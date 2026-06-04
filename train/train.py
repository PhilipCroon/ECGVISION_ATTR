#!/home/pmc57/miniconda3/envs/CiDATGAN/bin/python
"""
Train ECGVISION-ATTR (amyloid) on the difficulty-enriched matched cohort.

Ported from the existing PheWAS-style training script, specialized to a single
binary label (amyloid). Cohort source = tabs/train_matched_1_20.csv produced by
build_cohort.py + matching.R.

Label:  group == 'amyloid'  -> 1   (control / lvh / pyp_negative -> 0)
Key:    cohort carries FileID; training keys on fileID (FileID minus '.dcm').
Filter: format == 'full' (same as production script).
"""
# %%
import os
import sys
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageFile
from sklearn.utils import shuffle
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tqdm.keras import TqdmCallback

ImageFile.LOAD_TRUNCATED_IMAGES = True
pd.set_option('display.max_columns', 300)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project
from model import build_transfer_model, unfreeze_model, gc_callback
import model as _model_module

sys.path.append(project.ecg_image_models_path)
from utils import *           # noqa: F401,F403  (DataSequenceRAM, load_images_parallel, make_plot, ...)
from model_helpers import *   # noqa: F401,F403  (ContrastiveModel)

# === Config ===
LABEL = 'amyloid'
BATCH_SIZE = 256   # 128 per GPU across 2 GPUs; change back to 128 for single GPU
EPOCHS_FROZEN = 3       # head-only warmup
EPOCHS = 5              # full fine-tune after unfreeze
VAL_FRACTION = 0.15          # patient-level holdout from the matched train cohort
MAKE_IMAGE = True            # set True to precompute ECG images first
IMAGE_DIR = project.image_dir
MODEL_DIR = os.path.join(project.project_root, 'models')
FORMATS_FILE = project.formats_file
os.makedirs(MODEL_DIR, exist_ok=True)
save_date = datetime.today().strftime('%Y_%m_%d')

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPUs available: {len(gpus)}")

# Mixed precision: bfloat16 on H100 — faster, numerically same as float32
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# %% === Data ===
cohort = pd.read_csv(os.path.join(project.tabs_path, 'train_matched_1_20.csv'))

# Binary label
cohort[LABEL] = (cohort['group'] == 'amyloid').astype(np.float32)

# fileID key + format filter (mirror production script)
cohort['fileID'] = cohort['FileID'].astype(str).str.replace('.dcm$', '', regex=True)
formats = pd.read_csv(FORMATS_FILE).drop_duplicates(subset=['fileID'])
cohort = cohort.merge(formats[['fileID', 'format', 'format_new']], how='inner', on='fileID')
cohort = cohort[(cohort['format'] == 'full') & (cohort['format_new'] == 'full')]
cohort = cohort[cohort[LABEL].notna()].copy()

# Patient-level train/val split (no MRN leakage across the split)
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=15)
tr_idx, va_idx = next(gss.split(cohort, groups=cohort['MRN']))
train_df = shuffle(cohort.iloc[tr_idx], random_state=15)
validate_df = cohort.iloc[va_idx]
print(f"Train ECGs: {len(train_df)} ({train_df['MRN'].nunique()} MRNs)  "
      f"pos={int(train_df[LABEL].sum())}")
print(f"Val   ECGs: {len(validate_df)} ({validate_df['MRN'].nunique()} MRNs)  "
      f"pos={int(validate_df[LABEL].sum())}")

# Images -> RAM
if MAKE_IMAGE:
    print("🔁 Precomputing images...")
    save_all_images(train_df, IMAGE_DIR)
    save_all_images(validate_df, IMAGE_DIR)
print("Loading images from disk into RAM...")
train_images = load_images_parallel(train_df['fileID'], IMAGE_DIR)
train_df['image_array'] = train_df['fileID'].map(train_images)
train_df = train_df[train_df['image_array'].notna()].reset_index(drop=True)
assert len(train_df) > 0, (
    f"No images loaded from {IMAGE_DIR}. "
    "Set MAKE_IMAGE=True on first run to precompute images.")

val_images = load_images_parallel(validate_df['fileID'], IMAGE_DIR)
validate_df = validate_df.copy()
validate_df['image_array'] = validate_df['fileID'].map(val_images)
validate_df = validate_df[validate_df['image_array'].notna()].reset_index(drop=True)
assert len(validate_df) > 0, f"No validation images loaded from {IMAGE_DIR}."

# Effective number-of-samples class weights (data-driven, adapts to cohort imbalance)
_BETA = 0.999995
_n_pos = int(train_df[LABEL].sum())
_n_neg = len(train_df) - _n_pos
_w_pos = (1 - _BETA) / (1 - _BETA ** _n_pos)
_w_neg = (1 - _BETA) / (1 - _BETA ** _n_neg)
_model_module.CLASS_WEIGHTS = np.array([[_w_pos, _w_neg]])
print(f"Class weights — pos: {_w_pos:.5f}  neg: {_w_neg:.5f}  ratio: {_w_pos / _w_neg:.1f}x")

train_sequence = DataSequenceRAM(df=train_df, batch_size=BATCH_SIZE, label=LABEL)
validation_sequence = DataSequenceRAM(df=validate_df, batch_size=BATCH_SIZE, label=LABEL)

# %% === Train ===
frozen_model_file = os.path.join(MODEL_DIR, f'attr_{LABEL}_{save_date}_frozen' + '_{epoch:02d}')
checkpoint = ModelCheckpoint(frozen_model_file, monitor='val_auroc', mode='max',
                             save_best_only=True, verbose=1)
csv_logger = CSVLogger(os.path.join(MODEL_DIR, f'{LABEL}_{save_date}_trains.csv'),
                       append=True, separator=';')
early_stop_frozen = EarlyStopping(monitor='val_auroc', patience=2, mode='max', verbose=1)

# MirroredStrategy: single-node multi-GPU (uses all visible GPUs automatically)
strategy = tf.distribute.MirroredStrategy()
print(f"Training on {strategy.num_replicas_in_sync} GPU(s)")
num_workers = max(1, int(multiprocessing.cpu_count() * 0.5))

fit_kwargs = dict(
    validation_data=validation_sequence,
    verbose=0,
    callbacks=[TqdmCallback(verbose=1), checkpoint, csv_logger, gc_callback(), early_stop_frozen],
    use_multiprocessing=True,
    workers=num_workers,
    max_queue_size=16,
    shuffle=True,
)

# Phase 1: frozen encoder, head-only warmup
with strategy.scope():
    model_cnn = build_transfer_model()

print(f"Phase 1: frozen encoder, {EPOCHS_FROZEN} epochs @ LR={1e-3}")
model_cnn.fit(train_sequence, epochs=EPOCHS_FROZEN, **fit_kwargs)

# Phase 2: unfreeze all (except BN), fine-tune at lower LR
with strategy.scope():
    unfreeze_model(model_cnn)

saved_model_file = os.path.join(MODEL_DIR, f'attr_{LABEL}_{save_date}_unfrozen' + '_{epoch:02d}')
checkpoint = ModelCheckpoint(saved_model_file, monitor='val_auroc', mode='max',
                             save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_auroc', patience=3, mode='max', verbose=1)
fit_kwargs['callbacks'] = [TqdmCallback(verbose=1), checkpoint, csv_logger, gc_callback(), early_stop]

print(f"Phase 2: unfrozen, {EPOCHS} epochs @ LR=1e-5")
model_cnn.fit(train_sequence, epochs=EPOCHS, **fit_kwargs)

print("✅ Training complete.")
