"""
Evaluate a SEED-ENSEMBLE on the temporal holdout (cohort_test.csv): soft-average
the per-ECG probabilities of N checkpoints, then score single-models vs the
ensemble at the patient level, with a paired bootstrap CI of the gain.

Why soft-average (not weight-average): the members trained on different bagging
splits + seeds -> different loss basins, so only OUTPUTS may be averaged, never
weights. See train.py:71-83 and the discussion in eval.py.

Usage:
    python ensemble_eval.py                         # auto: best_ckpt per seed from run_registry.csv
    python ensemble_eval.py <ckpt1> <ckpt2> <ckpt3> # explicit checkpoint dirs
    python ensemble_eval.py --seeds 1 2 3           # pick these seeds from the registry

Picks, per seed, the LATEST run's best_ckpt (so a re-run supersedes an older one).
Renders the test cohort once (deterministic std layout), predicts each model one at
a time (never holds N models in VRAM), soft-averages, and reports:
  - per-model patient-level AUROC/AUPRC/Sens@90spec/Spec@90sens
  - ensemble patient-level metrics
  - paired bootstrap: ensemble AUROC, best-single AUROC, and Δ(ensemble - best single)
    with 95% CI. If the Δ-CI excludes 0, the ensemble gain is unlikely to be chance.

Saves: tabs/ensemble_eval_<date>.csv (per-ECG, one pred col per model + pred_ens)
       tabs/ensemble_metrics_<date>.csv (patient-level summary)
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project
from model import loss_fn
# reuse the validated single-model machinery so render/parse stays identical
from eval import (load_test_cohort, report_metrics, IMAGE_DIR, MODEL_DIR, LABEL)

sys.path.append(project.ecg_image_models_path)
from utils import load_images_parallel, save_all_images

REGISTRY_FILE = os.path.join(MODEL_DIR, 'run_registry.csv')
N_BOOT = 2000
BOOT_SEED = 20250910
ALPHA = 0.05


def _ci(vals):
    lo, hi = np.percentile(vals, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def resolve_checkpoints(explicit_paths, seeds):
    """Return [(label, abs_ckpt_path), ...]. Explicit paths win; else read registry."""
    if explicit_paths:
        out = []
        for p in explicit_paths:
            ap = p if os.path.isabs(p) else os.path.join(MODEL_DIR, p)
            assert os.path.isdir(ap), f"Checkpoint dir not found: {ap}"
            out.append((os.path.basename(ap.rstrip('/')), ap))
        return out

    assert os.path.exists(REGISTRY_FILE), (
        f"No {REGISTRY_FILE}. Pass checkpoint dirs explicitly.")
    reg = pd.read_csv(REGISTRY_FILE)
    reg = reg[reg['best_ckpt'].notna() & (reg['best_ckpt'].astype(str) != '')]
    if seeds:
        reg = reg[reg['seed'].isin([float(s) for s in seeds] + [int(s) for s in seeds])]
    assert len(reg) > 0, "No registry rows with a best_ckpt (did training finish?)."

    # per seed, keep the latest run (run_id is a sortable timestamp string)
    reg = reg.sort_values('run_id')
    picked = reg.groupby('seed', dropna=False).tail(1)
    out = []
    for _, row in picked.iterrows():
        ckpt = os.path.join(MODEL_DIR, str(row['best_ckpt']))
        if not os.path.isdir(ckpt):
            print(f"WARNING: best_ckpt missing on disk, skipping: {ckpt}")
            continue
        seed = row['seed']
        label = f"s{int(seed)}" if pd.notna(seed) else "noseed"
        out.append((label, ckpt))
    assert len(out) >= 2, (
        f"Need >=2 checkpoints to ensemble; resolved {len(out)}. "
        f"Check run_registry.csv best_ckpt paths.")
    return out


def predict_model(ckpt_path, X, batch_size=256):
    """Load one checkpoint, predict over X, free it. Returns 1-D prob array."""
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(ckpt_path, custom_objects={'loss_fn': loss_fn})
    preds = []
    n_batches = int(np.ceil(len(X) / batch_size))
    for i in tqdm(range(n_batches), desc=f"predict {os.path.basename(ckpt_path)}", unit="batch"):
        batch = X[i * batch_size:(i + 1) * batch_size]
        preds.append(model.predict_on_batch(batch))
    y = np.concatenate(preds).squeeze()
    del model
    tf.keras.backend.clear_session()
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='*', default=None,
                        help='Explicit checkpoint dirs (override registry auto-pick).')
    parser.add_argument('--seeds', nargs='*', default=None,
                        help='Restrict registry auto-pick to these seeds.')
    args = parser.parse_args()

    members = resolve_checkpoints(args.checkpoints, args.seeds)
    print("Ensemble members:")
    for label, path in members:
        print(f"  [{label}] {path}")

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # --- render + load test cohort ONCE (shared X across all members) ---
    cohort = load_test_cohort()
    print(f"Test ECGs: {len(cohort)} ({cohort['MRN'].nunique()} MRNs)  "
          f"pos={int(cohort[LABEL].sum())}")
    print("Rendering test images (deterministic std layout) once...")
    status = save_all_images(cohort, IMAGE_DIR, deterministic=True)
    print(f"  render status: {status}")
    images = load_images_parallel(cohort['fileID'], IMAGE_DIR)
    cohort['image_array'] = cohort['fileID'].map(images)
    cohort = cohort[cohort['image_array'].notna()].reset_index(drop=True)
    assert len(cohort) > 0, f"No images loaded from {IMAGE_DIR}"
    X = np.stack(cohort['image_array'].values)

    # --- per-member predictions (one model in VRAM at a time) ---
    member_cols = []
    for label, path in members:
        col = f"pred_{label}"
        # guard against duplicate labels (e.g. two noseed runs)
        if col in member_cols:
            col = f"pred_{label}_{len(member_cols)}"
        cohort[col] = predict_model(path, X)
        member_cols.append(col)

    # soft-average -> ensemble per-ECG probability
    cohort['pred_ens'] = cohort[member_cols].mean(axis=1)

    # --- patient-level aggregation (mean score per MRN), same as eval.py ---
    agg = {c: (c, 'mean') for c in member_cols + ['pred_ens']}
    agg['label'] = (LABEL, 'max')
    agg['group'] = ('group', 'first')
    pat = cohort.groupby('MRN').agg(**agg).reset_index()
    y = pat['label'].to_numpy()

    print("\n" + "=" * 60)
    print("PATIENT-LEVEL RESULTS (mean score per MRN)")
    print("=" * 60)
    summary = []
    for c in member_cols:
        r = report_metrics(y, pat[c].to_numpy(), label=f"single {c.replace('pred_', '')}")
        if r:
            r['model'] = c.replace('pred_', '')
            summary.append(r)
    r = report_metrics(y, pat['pred_ens'].to_numpy(), label="ENSEMBLE (soft-avg)")
    if r:
        r['model'] = 'ensemble'
        summary.append(r)

    # --- paired bootstrap: ensemble vs best single ---
    member_point = {c: roc_auc_score(y, pat[c].to_numpy()) for c in member_cols}
    best_col = max(member_point, key=member_point.get)
    best_label = best_col.replace('pred_', '')
    print("\n" + "=" * 60)
    print(f"PAIRED BOOTSTRAP  (n_pat={len(y)}, pos={int(y.sum())}, {N_BOOT} boots)")
    print(f"best single member on test = {best_label} (AUROC={member_point[best_col]:.4f})")
    print("=" * 60)

    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    idx_sets = []
    while len(idx_sets) < N_BOOT:
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            idx_sets.append(idx)

    s_ens = pat['pred_ens'].to_numpy()
    s_best = pat[best_col].to_numpy()
    b_ens = np.array([roc_auc_score(y[i], s_ens[i]) for i in idx_sets])
    b_best = np.array([roc_auc_score(y[i], s_best[i]) for i in idx_sets])
    elo, ehi = _ci(b_ens)
    blo, bhi = _ci(b_best)
    diff = b_ens - b_best           # paired: same resample indices
    dlo, dhi = _ci(diff)
    p_gt = float(np.mean(diff > 0))
    verdict = ("ENSEMBLE > best single (CI excludes 0)" if dlo > 0 else
               "best single > ensemble (CI excludes 0)" if dhi < 0 else
               "overlaps 0 — gain NOT distinguishable from noise")
    print(f"  ensemble    AUROC = {roc_auc_score(y, s_ens):.4f}  [{elo:.4f}, {ehi:.4f}]")
    print(f"  best single AUROC = {roc_auc_score(y, s_best):.4f}  [{blo:.4f}, {bhi:.4f}]")
    print(f"  Δ (ensemble - {best_label}) = {diff.mean():+.4f}  [{dlo:+.4f}, {dhi:+.4f}]  "
          f"P(ens>best)={p_gt:.2f}")
    print(f"  -> {verdict}")

    # --- save ---
    save_date = datetime.today().strftime('%Y_%m_%d')
    keep = ['MRN', 'fileID', LABEL, 'group'] + member_cols + ['pred_ens']
    keep = [c for c in keep if c in cohort.columns]
    out_path = os.path.join(project.tabs_path, f'ensemble_eval_{save_date}.csv')
    cohort[keep].to_csv(out_path, index=False)
    metrics_path = os.path.join(project.tabs_path, f'ensemble_metrics_{save_date}.csv')
    pd.DataFrame(summary).to_csv(metrics_path, index=False)
    print(f"\nSaved per-ECG preds: {out_path}")
    print(f"Saved metrics:       {metrics_path}")


if __name__ == '__main__':
    main()
