"""
Fast smoke test: build model, pull one augmented batch, run one train step.
Catches dtype/shape/policy crashes in ~30s before a full overnight run.

Run on the server:  python smoke_test.py
Exit code 0 = safe to launch training. Non-zero = fix before running.
"""
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))
import project_constants as project

# Mirror train.py exactly: same global policy is what triggers dtype bugs
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global policy: {tf.keras.mixed_precision.global_policy().name}")

from utils import DataSequenceAugRAM, DataSequenceRAM
from model import build_transfer_model, unfreeze_model

LABEL = 'amyloid'
BATCH_SIZE = 8          # tiny — we only care about code paths, not throughput
N = 24                  # 3 batches

# --- synthetic [0,1] images, shape (300,300,3), mixed labels ---
print("\nBuilding synthetic batch...")
rng = np.random.RandomState(0)
df = pd.DataFrame({
    LABEL: np.array([1, 0, 0, 0, 0, 0, 0, 0] * (N // 8), dtype=np.float32),
})
df['image_array'] = [rng.rand(300, 300, 3).astype(np.float32) for _ in range(N)]

# --- check augmented sequence output ---
print("Pulling one augmented batch from DataSequenceAugRAM...")
train_seq = DataSequenceAugRAM(df=df, batch_size=BATCH_SIZE, label=LABEL)
bx, by = train_seq[0]
print(f"  batch_x: shape={bx.shape}  dtype={bx.dtype}  "
      f"min={bx.min():.3f}  max={bx.max():.3f}")
print(f"  batch_y: shape={by.shape}  dtype={by.dtype}")
assert bx.shape == (BATCH_SIZE, 300, 300, 3), f"bad x shape {bx.shape}"
assert by.shape == (BATCH_SIZE, 1), f"bad y shape {by.shape}"
assert 0.0 <= bx.min() and bx.max() <= 1.5, f"scale off: [{bx.min()},{bx.max()}]"

# --- validation sequence (no aug) ---
val_seq = DataSequenceRAM(df=df, batch_size=BATCH_SIZE, label=LABEL)
vbx, vby = val_seq[0]
assert vbx.shape == (BATCH_SIZE, 300, 300, 3), f"bad val x shape {vbx.shape}"

# --- build model + one frozen-phase train step ---
print("\nBuilding model (frozen encoder)...")
model = build_transfer_model()
print("Running 1 frozen train step...")
logs = model.train_on_batch(bx, by, return_dict=True)
print(f"  frozen step OK: {logs}")
assert np.isfinite(logs['loss']), "non-finite loss in frozen step"

# --- unfreeze + one fine-tune step (this is where OOM/dtype bugs hide) ---
print("\nUnfreezing + running 1 fine-tune step...")
unfreeze_model(model)
logs2 = model.train_on_batch(bx, by, return_dict=True)
print(f"  unfrozen step OK: {logs2}")
assert np.isfinite(logs2['loss']), "non-finite loss in unfrozen step"

# --- predict path (mirrors val/eval) ---
print("\nRunning predict...")
preds = model.predict(vbx, verbose=0)
print(f"  preds: shape={preds.shape}  dtype={preds.dtype}  "
      f"range=[{preds.min():.3f},{preds.max():.3f}]")
assert preds.shape == (BATCH_SIZE, 1), f"bad pred shape {preds.shape}"
assert np.all(np.isfinite(preds)), "non-finite predictions"

print("\n✅ SMOKE TEST PASSED — safe to launch training.")
