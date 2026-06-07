"""
Evaluate ECGVISION-ATTR on the temporal holdout (cohort_test.csv).

Usage:
    python eval.py <model_path>    # explicit checkpoint path
    python eval.py                 # auto-picks latest unfrozen checkpoint in MODEL_DIR

Reports patient-level AUROC / AUPRC / Sens@90%spec / Spec@90%sens for:
  - All patients
  - Mimics subset (LVH/AS/Amyloid — hardest false-positive cases)
  - In PyP (referred to CARI scan)

Saves per-ECG predictions to tabs/eval_<date>.csv.
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project
from model import loss_fn

sys.path.append(project.ecg_image_models_path)
from utils import load_images_parallel

LABEL = 'amyloid'
IMAGE_DIR = project.image_dir
MODEL_DIR = os.path.join(project.project_root, 'models')
FORMATS_FILE = project.formats_file


def load_test_cohort(assume_full=True):
    """
    Load the temporal-holdout test cohort.

    assume_full=True (default): skip the formats_rerun.csv lookup and treat every
      ECG as 'full' layout. The new (post-cutoff) ECGs use a different fileID
      scheme (/data/ecg/<ts>_<hex>.xml) than formats_rerun.csv (<yyyy_mm>/<id>...),
      so the lookup returns nothing for them. New PyP-workup ECGs are standard
      12-leads, so 'full' is the right default. Set False to use the lookup+filter.
    """
    path = os.path.join(project.tabs_path, 'cohort_test.csv')
    assert os.path.exists(path), f"cohort_test.csv not found at {path} — run src/build_cohort.py"
    cohort = pd.read_csv(path)
    print(f"[load_test_cohort] raw rows: {len(cohort)}")
    if 'group' in cohort.columns:
        print(f"[load_test_cohort] group: {cohort['group'].value_counts().to_dict()}")

    cohort[LABEL] = (cohort['group'] == 'amyloid').astype(np.float32)
    # New 2025 ECGs use full paths like /data/ecg/<ts>_<hex>.xml; old ones are bare
    # stems. Reduce both to the bare stem make_plot expects (basename, no extension).
    # Matches AUMC_CMR prepare_ecg2cmr_inputs.py line 254-257.
    cohort['fileID'] = cohort['FileID'].astype(str).map(
        lambda f: os.path.splitext(os.path.basename(f))[0])
    print(f"[load_test_cohort] sample cleaned fileID: {cohort['fileID'].head(2).tolist()}")

    if assume_full:
        cohort['format'] = 'full'
        cohort['format_new'] = 'full'
        print(f"[load_test_cohort] assume_full=True -> all {len(cohort)} ECGs treated as 'full'")
    else:
        formats = pd.read_csv(FORMATS_FILE).drop_duplicates(subset=['fileID'])
        cohort = cohort.merge(formats[['fileID', 'format', 'format_new']], how='inner', on='fileID')
        print(f"[load_test_cohort] after format merge: {len(cohort)}")
        cohort = cohort[(cohort['format'] == 'full') & (cohort['format_new'] == 'full')]
        print(f"[load_test_cohort] after format=='full' filter: {len(cohort)}")

    cohort = cohort[cohort[LABEL].notna()].copy()
    if len(cohort) == 0:
        raise ValueError("Test cohort is empty. With assume_full=False this means a "
                         "fileID mismatch vs formats_rerun.csv (new ECGs not covered).")
    return cohort


def find_latest_checkpoint():
    entries = [
        os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR)
        if 'unfrozen' in d and os.path.isdir(os.path.join(MODEL_DIR, d))
    ]
    if not entries:
        raise FileNotFoundError(f"No unfrozen checkpoints in {MODEL_DIR}")
    return max(entries, key=os.path.getmtime)


def sensitivity_at_specificity(y_true, y_score, target_spec=0.90):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    best = idx[np.argmin(spec[idx] - target_spec)]
    return float(tpr[best]), float(thresholds[best])


def specificity_at_sensitivity(y_true, y_score, target_sens=0.90):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.where(tpr >= target_sens)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    best = idx[0]  # first (highest-specificity) point where sens >= target
    return float(1 - fpr[best]), float(thresholds[best])


def report_metrics(y_true, y_score, label):
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        print(f"\n{label}  — skipped (n={len(y_true)}, pos={int(y_true.sum())})")
        return {}
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    sens, thr_sens = sensitivity_at_specificity(y_true, y_score, 0.90)
    spec, thr_spec = specificity_at_sensitivity(y_true, y_score, 0.90)
    print(f"\n{label}  (n={len(y_true)}, pos={int(y_true.sum())})")
    print(f"  AUROC:         {auroc:.4f}")
    print(f"  AUPRC:         {auprc:.4f}")
    print(f"  Sens@90%spec:  {sens:.4f}  (thr={thr_sens:.4f})")
    print(f"  Spec@90%sens:  {spec:.4f}  (thr={thr_spec:.4f})")
    return {'label': label, 'n': len(y_true), 'pos': int(y_true.sum()),
            'auroc': auroc, 'auprc': auprc,
            'sens_90spec': sens, 'thr_sens': thr_sens,
            'spec_90sens': spec, 'thr_spec': thr_spec}


def patient_level(ecg_df, score_col='pred', label_col=LABEL):
    return (ecg_df.groupby('MRN')
            .agg(pred_mean=(score_col, 'mean'),
                 pred_max=(score_col, 'max'),
                 label=(label_col, 'max'),
                 group=('group', 'first'),
                 Composite_LVH_binary=('Composite_LVH_binary', 'max') if 'Composite_LVH_binary' in ecg_df.columns else ('MRN', 'count'),
                 SevereAS_ObjSubj=('SevereAS_ObjSubj', 'max') if 'SevereAS_ObjSubj' in ecg_df.columns else ('MRN', 'count'))
            .reset_index())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', default=None)
    args = parser.parse_args()

    model_path = args.model_path or find_latest_checkpoint()
    print(f"Model: {model_path}")

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    model = tf.keras.models.load_model(
        model_path, custom_objects={'loss_fn': loss_fn})

    cohort = load_test_cohort()
    print(f"Test ECGs: {len(cohort)} ({cohort['MRN'].nunique()} MRNs)  "
          f"pos={int(cohort[LABEL].sum())}")
    print(f"Groups: {cohort['group'].value_counts().to_dict()}")

    print("Loading test images...")
    images = load_images_parallel(cohort['fileID'], IMAGE_DIR)
    cohort['image_array'] = cohort['fileID'].map(images)
    cohort = cohort[cohort['image_array'].notna()].reset_index(drop=True)
    assert len(cohort) > 0, f"No images loaded from {IMAGE_DIR}"

    X = np.stack(cohort['image_array'].values)
    print("Predicting...")
    batch_size = 256
    n_batches = int(np.ceil(len(X) / batch_size))
    preds = []
    for i in tqdm(range(n_batches), desc="Predicting", unit="batch"):
        batch = X[i * batch_size:(i + 1) * batch_size]
        preds.append(model.predict_on_batch(batch))
    y_pred = np.concatenate(preds).squeeze()
    cohort['pred'] = y_pred

    # --- Patient-level aggregation ---
    pat = (cohort.groupby('MRN')
           .agg(pred_mean=('pred', 'mean'),
                pred_max=('pred', 'max'),
                label=(LABEL, 'max'),
                group=('group', 'first'))
           .reset_index())

    # Carry LVH/AS flags to patient level if present
    for col in ('Composite_LVH_binary', 'SevereAS_ObjSubj'):
        if col in cohort.columns:
            pat = pat.merge(
                cohort.groupby('MRN')[col].max().reset_index(), on='MRN', how='left')

    # --- PyP referred mask ---
    pyp_mrns = set()
    if os.path.exists(project.pyp_file):
        try:
            pyp_df = pd.read_parquet(project.pyp_file, columns=['MRN'])
            pyp_mrns = set(pyp_df['MRN'].dropna().astype(str))
        except Exception as e:
            print(f"Warning: could not load PyP file: {e}")
    pat['in_pyp'] = pat['MRN'].astype(str).isin(pyp_mrns)

    # --- Mimics mask: amyloid cases + LVH/AS patients ---
    if 'Composite_LVH_binary' in pat.columns and 'SevereAS_ObjSubj' in pat.columns:
        mimics_mask = (pat['label'] == 1) | \
                      (pat['Composite_LVH_binary'].fillna(0) == 1) | \
                      (pat['SevereAS_ObjSubj'].fillna(0) == 1)
    else:
        mimics_mask = pat['label'] == 1  # fallback: amyloid only
        print("Warning: LVH/AS columns not found — mimics subset = amyloid only")

    results = []

    print("\n" + "=" * 60)
    print("PATIENT-LEVEL RESULTS (mean score per MRN)")
    print("=" * 60)

    # All patients
    results.append(report_metrics(pat['label'].values, pat['pred_mean'].values,
                                  label='All patients'))

    # Mimics subset
    m = pat[mimics_mask]
    results.append(report_metrics(m['label'].values, m['pred_mean'].values,
                                  label='Mimics (LVH/AS/Amyloid)'))

    # PyP referred
    p = pat[pat['in_pyp']]
    results.append(report_metrics(p['label'].values, p['pred_mean'].values,
                                  label='In PyP (CARI referred)'))

    # Not in PyP
    np_ = pat[~pat['in_pyp']]
    results.append(report_metrics(np_['label'].values, np_['pred_mean'].values,
                                  label='Not in PyP'))

    # Age > 65
    if 'Age' in cohort.columns:
        pat_age = pat.merge(
            cohort.groupby('MRN')['Age'].first().reset_index(), on='MRN', how='left')
        over65 = pat_age[pat_age['Age'] > 65]
        results.append(report_metrics(over65['label'].values, over65['pred_mean'].values,
                                      label='Age > 65'))

    # Save predictions
    save_date = datetime.today().strftime('%Y_%m_%d')
    out_path = os.path.join(project.tabs_path, f'eval_{save_date}.csv')
    cohort[['MRN', 'fileID', LABEL, 'group', 'pred']].to_csv(out_path, index=False)

    results_path = os.path.join(project.tabs_path, f'eval_metrics_{save_date}.csv')
    pd.DataFrame([r for r in results if r]).to_csv(results_path, index=False)
    print(f"\nSaved predictions: {out_path}")
    print(f"Saved metrics:     {results_path}")


if __name__ == '__main__':
    main()
