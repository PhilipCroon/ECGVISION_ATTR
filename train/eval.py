"""
Evaluate ECGVISION-ATTR on the temporal holdout (cohort_test.csv).

Usage:
    python eval.py <model_path>    # explicit checkpoint path
    python eval.py                 # auto-picks latest unfrozen checkpoint in MODEL_DIR

Reports ECG-level and patient-level AUROC / AUPRC / Sens@90%spec.
Saves per-ECG predictions to tabs/eval_<date>.csv.
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
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


def load_test_cohort():
    path = os.path.join(project.tabs_path, 'cohort_test.csv')
    cohort = pd.read_csv(path)
    cohort[LABEL] = (cohort['group'] == 'amyloid').astype(np.float32)
    cohort['fileID'] = cohort['FileID'].astype(str).str.replace('.dcm$', '', regex=True)
    formats = pd.read_csv(FORMATS_FILE).drop_duplicates(subset=['fileID'])
    cohort = cohort.merge(formats[['fileID', 'format', 'format_new']], how='inner', on='fileID')
    cohort = cohort[(cohort['format'] == 'full') & (cohort['format_new'] == 'full')]
    return cohort[cohort[LABEL].notna()].copy()


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


def report_metrics(y_true, y_score, label):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    sens, thresh = sensitivity_at_specificity(y_true, y_score, 0.90)
    print(f"\n{label}  (n={len(y_true)}, pos={int(y_true.sum())})")
    print(f"  AUROC:         {auroc:.4f}")
    print(f"  AUPRC:         {auprc:.4f}")
    print(f"  Sens@90%spec:  {sens:.4f}  (threshold={thresh:.4f})")
    return {'auroc': auroc, 'auprc': auprc, 'sens_90spec': sens, 'threshold': thresh}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', default=None,
                        help='Path to saved model checkpoint')
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
    y_pred = model.predict(X, batch_size=256, verbose=1).squeeze()
    cohort['pred'] = y_pred

    # ECG-level
    report_metrics(cohort[LABEL].values, y_pred, label='ECG-level')

    # Patient-level (mean and max aggregation)
    pat = (cohort.groupby('MRN')
           .agg(pred_mean=('pred', 'mean'),
                pred_max=('pred', 'max'),
                label=(LABEL, 'max'))
           .reset_index())
    report_metrics(pat['label'].values, pat['pred_mean'].values, label='Patient-level (mean)')
    report_metrics(pat['label'].values, pat['pred_max'].values,  label='Patient-level (max)')

    save_date = datetime.today().strftime('%Y_%m_%d')
    out_path = os.path.join(project.tabs_path, f'eval_{save_date}.csv')
    cohort[['MRN', 'fileID', LABEL, 'group', 'pred']].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
