#!/home/pmc57/miniconda3/envs/CiDATGAN/bin/python
"""
Train ECGVISION-ATTR (amyloid) on the difficulty-enriched matched cohort.

Ported from the existing PheWAS-style training script, specialized to a single
binary label (amyloid). Cohort source = tabs/train_matched_1_10.csv produced by
build_cohort.py + matching.R.

Label:  group == 'amyloid'  -> 1   (control / lvh / pyp_negative -> 0)
Key:    cohort carries FileID; training keys on fileID (FileID minus '.dcm').
Filter: format == 'full' (same as production script).
"""
# %%
import csv
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
from tensorflow.keras.callbacks import (ModelCheckpoint, CSVLogger, EarlyStopping,
                                        ReduceLROnPlateau)
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
BATCH_SIZE = 256   # fits on a dedicated H100 80GB; drop to 128 if sharing the GPU
EPOCHS_FROZEN = 3       # head-only warmup
EPOCHS = 20             # full fine-tune after unfreeze
VAL_FRACTION = 0.15          # patient-level holdout from the matched train cohort
MAKE_IMAGE = True            # set True to precompute ECG images first
IMAGE_DIR = project.image_dir
MODEL_DIR = os.path.join(project.project_root, 'models')
FORMATS_FILE = project.formats_file
os.makedirs(MODEL_DIR, exist_ok=True)
save_date = datetime.today().strftime('%Y_%m_%d')
run_id = datetime.today().strftime('%Y%m%d_%H%M%S')


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


GIT_SHA = _git_sha()
RUN_TAG = os.getenv('RUN_TAG', '')   # free-text: what you changed this run, e.g. RUN_TAG="focal_loss"

# SEED: set for ensemble-diverse runs. Seeds python/numpy/TF RNGs (head init, dropout,
# aug ordering) AND the train/val split (SPLIT_STATE below) -> each member trains on a
# different subset, bagging-style. Safe because the internal TEST is a disjoint temporal
# holdout. Seed is appended to run_id so the 3 checkpoints don't collide.
import random
_seed_env = os.getenv('SEED', '')
SEED = int(_seed_env) if _seed_env.strip() else None
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    run_id = f"{run_id}_s{SEED}"
    print(f"SEED={SEED} -> run_id={run_id}")

# === Run registry ===
REGISTRY_FILE = os.path.join(MODEL_DIR, 'run_registry.csv')
REGISTRY_FIELDS = [
    'run_id', 'date', 'git_sha', 'run_tag', 'seed', 'label',
    'loss_fn', 'class_weights', 'match_ratio', 'data_key', 'augmentation',
    'batch_size', 'epochs_frozen', 'epochs_unfrozen', 'val_fraction',
    'train_sequence', 'lr_frozen', 'lr_unfrozen', 'mixed_precision',
    'n_train', 'n_val', 'n_train_pos', 'n_val_pos',
    'n_train_indiv', 'n_val_indiv', 'n_train_pos_indiv', 'n_val_pos_indiv',
    'best_val_auroc_frozen', 'best_val_auroc_unfrozen', 'best_epoch_unfrozen',
    'best_ckpt', 'epoch_csv',
]

def _write_registry(row: dict):
    write_header = not os.path.exists(REGISTRY_FILE)
    with open(REGISTRY_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class RegistryCallback(tf.keras.callbacks.Callback):
    """Writes best val_auroc to run_registry.csv at end of phase 2."""
    def __init__(self, registry_row: dict, ckpt_template: str = None):
        super().__init__()
        self.registry_row = registry_row
        self.ckpt_template = ckpt_template   # e.g. ".../attr_amyloid_<run_id>_unfrozen_{epoch:02d}"
        self.best_val_auroc = -np.inf
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        val_auroc = (logs or {}).get('val_auroc', -np.inf)
        if val_auroc > self.best_val_auroc:
            self.best_val_auroc = val_auroc
            self.best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        self.registry_row['best_val_auroc_unfrozen'] = round(self.best_val_auroc, 5)
        self.registry_row['best_epoch_unfrozen'] = self.best_epoch
        if self.ckpt_template and self.best_epoch > 0:
            self.registry_row['best_ckpt'] = os.path.basename(
                self.ckpt_template.format(epoch=self.best_epoch))
        _write_registry(self.registry_row)
        print(f"\nRun registry updated: {REGISTRY_FILE}")
        print(f"  run_id={self.registry_row['run_id']}  "
              f"best_val_auroc={self.best_val_auroc:.4f}  epoch={self.best_epoch}")


# GPU setup
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPUs available: {len(gpus)}")

# Mixed precision: bfloat16 on H100 — faster, numerically same as float32
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# %% === Data ===
cohort = pd.read_csv(os.path.join(project.tabs_path, 'train_matched_1_10.csv'))

# Binary label
cohort[LABEL] = (cohort['group'] == 'amyloid').astype(np.float32)

# fileID key: 'fileID' from the ECG metadata is the store-relative path (e.g.
# pdcfs1/2024/03/..., ekg_waveform_2021/..., or a bare stem for old ECGs). make_plot
# loads all_ecgs/<fileID>.npy, so the subdir prefix MUST be kept. Do NOT basename it.
# Fall back to basename(FileID) only if the relative fileID column is absent.
if 'fileID' in cohort.columns and cohort['fileID'].notna().any():
    cohort['fileID'] = cohort['fileID'].astype(str)
else:
    cohort['fileID'] = cohort['FileID'].astype(str).map(
        lambda f: os.path.splitext(os.path.basename(f))[0])

# format filter: old ECGs are looked up in formats_rerun.csv and kept only if
# format/format_new == 'full'. New ECGs are NOT in that file (different fileID
# scheme); they are standard PyP-workup 12-leads, so LEFT-merge + fill 'full'
# instead of dropping them with an inner join. The n_new print is the tripwire:
# it should be ~the number of post-update ECGs, NOT wildly larger. If large,
# formats_rerun.csv is patchy and old ECGs of unknown layout are being assumed full.
formats = pd.read_csv(FORMATS_FILE).drop_duplicates(subset=['fileID'])
n_before = len(cohort)
cohort = cohort.merge(formats[['fileID', 'format', 'format_new']], how='left', on='fileID')
n_new = int(cohort['format'].isna().sum())
print(f"[train] {n_new}/{n_before} ECGs not in formats_rerun.csv -> assumed 'full' (new scheme)")
cohort['format'] = cohort['format'].fillna('full')
cohort['format_new'] = cohort['format_new'].fillna('full')
cohort = cohort[(cohort['format'] == 'full') & (cohort['format_new'] == 'full')]
cohort = cohort[cohort[LABEL].notna()].copy()

# Patient-level train/val split (no MRN leakage across the split).
# SPLIT_STATE varies with SEED so each ensemble member trains on a different subset
# (bagging-style -> more decorrelated members, collectively uses all the train pool).
# Safe because the internal TEST (cohort_test.csv) is a disjoint temporal holdout, so
# it never leaks into any member. Per-run val sets DIFFER -> judge the ensemble on the
# held-out test/external cohorts, not on val_auroc.
SPLIT_STATE = SEED if SEED is not None else 15
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SPLIT_STATE)
tr_idx, va_idx = next(gss.split(cohort, groups=cohort['MRN']))
train_df = shuffle(cohort.iloc[tr_idx], random_state=SPLIT_STATE)
validate_df = cohort.iloc[va_idx]
print(f"Train ECGs: {len(train_df)} ({train_df['MRN'].nunique()} MRNs)  "
      f"pos={int(train_df[LABEL].sum())}")
print(f"Val   ECGs: {len(validate_df)} ({validate_df['MRN'].nunique()} MRNs)  "
      f"pos={int(validate_df[LABEL].sum())}")

# Images -> RAM from PNG disk cache (fast; augmentation applied per-batch at train time)
if MAKE_IMAGE:
    print("🔁 Precomputing images...")
    save_all_images(train_df, IMAGE_DIR)
    save_all_images(validate_df, IMAGE_DIR)
print("Loading images from disk into RAM...")
train_images = load_images_parallel(train_df['fileID'], IMAGE_DIR)
train_df['image_array'] = train_df['fileID'].map(train_images)
train_df = train_df[train_df['image_array'].notna()].reset_index(drop=True)
assert len(train_df) > 0, (
    f"No images loaded from {IMAGE_DIR}. Set MAKE_IMAGE=True on first run.")
val_images = load_images_parallel(validate_df['fileID'], IMAGE_DIR)
validate_df = validate_df.copy()
validate_df['image_array'] = validate_df['fileID'].map(val_images)
validate_df = validate_df[validate_df['image_array'].notna()].reset_index(drop=True)
assert len(validate_df) > 0, f"No validation images loaded from {IMAGE_DIR}."

# --- actual usable data after rendering/loading (images, not just rows) ---
def _summarize(df, name):
    is_case = df[LABEL] == 1
    cases, controls = df[is_case], df[~is_case]
    print(f"[{name}] images: {len(df)}  individuals: {df['MRN'].nunique()}")
    print(f"    cases    -> images: {len(cases):6d}  individuals: {cases['MRN'].nunique()}")
    print(f"    controls -> images: {len(controls):6d}  individuals: {controls['MRN'].nunique()}")

print("\n=== Actual training data (images successfully rendered + loaded) ===")
_summarize(train_df, "TRAIN")
_summarize(validate_df, "VAL")
print("=" * 64 + "\n")

# Mild class weights — cohort is already 1:10 matched, so imbalance is handled at data level
_model_module.CLASS_WEIGHTS = np.array([[1.3, 0.77]])
print("Class weights — pos: 1.30  neg: 0.77  ratio: 1.7x")

AUG_MODE = os.getenv('AUG', 'rotate')   # 'rotate' (default) | 'scan' (scan/print artifacts)
print(f"Augmentation mode: {AUG_MODE}")
train_sequence = DataSequenceAugRAM(df=train_df, batch_size=BATCH_SIZE, label=LABEL,
                                    aug_mode=AUG_MODE)
validation_sequence = DataSequenceRAM(df=validate_df, batch_size=BATCH_SIZE, label=LABEL)

# One-time aug preview: dump a few augmented training images so the scan aug can be
# eyeballed before committing a full run. Should still read as a legible 12-lead.
if AUG_MODE == 'scan':
    import PIL.Image as _PImg
    _prev_dir = os.path.join(project.tabs_path, 'aug_preview')
    os.makedirs(_prev_dir, exist_ok=True)
    _bx, _ = train_sequence[0]
    for _i in range(min(6, len(_bx))):
        _PImg.fromarray((np.clip(_bx[_i], 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(_prev_dir, f'aug_{_i}.png'))
    print(f"Saved {min(6, len(_bx))} scan-aug previews -> {_prev_dir} (EYEBALL before trusting run)")

# %% === Train ===
epoch_csv = os.path.join(MODEL_DIR, f'{LABEL}_{run_id}_epochs.csv')

frozen_model_file = os.path.join(MODEL_DIR, f'attr_{LABEL}_{run_id}_frozen' + '_{epoch:02d}')
checkpoint = ModelCheckpoint(frozen_model_file, monitor='val_auroc', mode='max',
                             save_best_only=True, verbose=1)
csv_logger = CSVLogger(epoch_csv, append=True, separator=';')
early_stop_frozen = EarlyStopping(monitor='val_auroc', patience=2, mode='max', verbose=1)

# MirroredStrategy for multi-GPU; fall back to default for single GPU
# (MirroredStrategy triggers NCCL init even on 1 GPU, which can crash on some H100 configs).
# Use HierarchicalCopyAllReduce instead of the default NCCL all-reduce: NCCL fails with
# "unhandled system error / Adam/NcclAllReduce" on this box's 2-GPU setup.
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
else:
    strategy = tf.distribute.get_strategy()  # default single-device strategy
print(f"Training on {strategy.num_replicas_in_sync} GPU(s)")
num_workers = max(1, int(multiprocessing.cpu_count() * 0.25))

fit_kwargs = dict(
    validation_data=validation_sequence,
    verbose=0,
    callbacks=[TqdmCallback(verbose=1), checkpoint, csv_logger, gc_callback(), early_stop_frozen],
    use_multiprocessing=False,
    workers=num_workers,
    shuffle=True,
)

# Phase 1: frozen encoder, head-only warmup
with strategy.scope():
    model_cnn = build_transfer_model()

print(f"Phase 1: frozen encoder, {EPOCHS_FROZEN} epochs @ LR={1e-3}")
history_frozen = model_cnn.fit(train_sequence, epochs=EPOCHS_FROZEN, **fit_kwargs)
best_val_auroc_frozen = max(history_frozen.history.get('val_auroc', [-1]))

# Phase 2: unfreeze all (except BN), fine-tune at lower LR
with strategy.scope():
    unfreeze_model(model_cnn)

saved_model_file = os.path.join(MODEL_DIR, f'attr_{LABEL}_{run_id}_unfrozen' + '_{epoch:02d}')
checkpoint = ModelCheckpoint(saved_model_file, monitor='val_auroc', mode='max',
                             save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_auroc', patience=8, mode='max', verbose=1)
# Start hot (LR_UNFROZEN=1e-4), drop LR when val_auroc stalls -> fine-settle.
reduce_lr = ReduceLROnPlateau(monitor='val_auroc', mode='max', factor=0.5,
                              patience=2, min_lr=1e-6, verbose=1)

registry_row = dict(
    run_id=run_id,
    date=save_date,
    git_sha=GIT_SHA,
    run_tag=RUN_TAG,
    seed=SEED,
    label=LABEL,
    loss_fn=(f'focal(gamma={_model_module.FOCAL_GAMMA},alpha={_model_module.FOCAL_ALPHA})'
             if _model_module.LOSS == 'focal' else 'weighted_bce'),
    class_weights=str(_model_module.CLASS_WEIGHTS.tolist()),
    match_ratio=getattr(project, 'MATCH_RATIO', None),
    data_key=os.path.basename(project.ecg_metadata_file),
    augmentation=('rotation+-10deg+scan(zoom,translate,grid,bright,contrast,gamma,noise,blur,jpeg)'
                  if AUG_MODE == 'scan' else 'rotation+-10deg'),
    batch_size=BATCH_SIZE,
    epochs_frozen=EPOCHS_FROZEN,
    epochs_unfrozen=EPOCHS,
    val_fraction=VAL_FRACTION,
    train_sequence='DataSequenceAugRAM',
    lr_frozen=1e-3,
    lr_unfrozen='1e-4 + ReduceLROnPlateau(factor=0.5,patience=2,min_lr=1e-6)',
    mixed_precision='mixed_bfloat16',
    n_train=len(train_df),
    n_val=len(validate_df),
    n_train_pos=int(train_df[LABEL].sum()),
    n_val_pos=int(validate_df[LABEL].sum()),
    n_train_indiv=int(train_df['MRN'].nunique()),
    n_val_indiv=int(validate_df['MRN'].nunique()),
    n_train_pos_indiv=int(train_df[train_df[LABEL] == 1]['MRN'].nunique()),
    n_val_pos_indiv=int(validate_df[validate_df[LABEL] == 1]['MRN'].nunique()),
    best_val_auroc_frozen=round(best_val_auroc_frozen, 5),
    best_val_auroc_unfrozen=None,
    best_epoch_unfrozen=None,
    best_ckpt=None,
    epoch_csv=os.path.basename(epoch_csv),
)
registry_cb = RegistryCallback(registry_row, ckpt_template=saved_model_file)

fit_kwargs['callbacks'] = [TqdmCallback(verbose=1), checkpoint, csv_logger, gc_callback(),
                           reduce_lr, early_stop, registry_cb]

print(f"Phase 2: unfrozen, {EPOCHS} epochs @ LR=ExponentialDecay(2e-5), augmentation=rotation±10°")
model_cnn.fit(train_sequence, epochs=EPOCHS, **fit_kwargs)

print("✅ Training complete.")
