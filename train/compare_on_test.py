"""
Evaluate one or more model checkpoints on the SAME test set and print a
side-by-side AUROC/AUPRC comparison. Use to check if a new model beats an old one.

Usage:
    python compare_on_test.py <model_a> [<model_b> ...]
    python compare_on_test.py                          # auto: latest unfrozen checkpoint

Loads test images once, predicts each model, aggregates patient-level (mean score
per MRN). Test set = tabs/cohort_test.csv (never seen during training).
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project
from model import loss_fn
from eval import (load_test_cohort, find_latest_checkpoint,
                  sensitivity_at_specificity, specificity_at_sensitivity)

sys.path.append(project.ecg_image_models_path)
from utils import load_images_parallel, save_all_images

LABEL = 'amyloid'
IMAGE_DIR = project.image_dir
MODEL_DIR = os.path.join(project.project_root, 'models')


def predict(model, X, batch_size=256):
    preds = []
    n = int(np.ceil(len(X) / batch_size))
    for i in tqdm(range(n), desc="  predicting", unit="batch", leave=False):
        preds.append(model.predict_on_batch(X[i * batch_size:(i + 1) * batch_size]))
    return np.concatenate(preds).squeeze()


def patient_level(cohort):
    return (cohort.groupby('MRN')
            .agg(pred_mean=('pred', 'mean'),
                 label=(LABEL, 'max'),
                 group=('group', 'first'))
            .reset_index())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', help='checkpoint paths (default: latest unfrozen)')
    args = parser.parse_args()

    model_paths = args.models or [find_latest_checkpoint()]
    print(f"Comparing {len(model_paths)} model(s) on test set:")
    for p in model_paths:
        print(f"  - {p}")

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # --- load test cohort + images ONCE ---
    cohort = load_test_cohort()
    print(f"\nTest ECGs: {len(cohort)} ({cohort['MRN'].nunique()} MRNs)  "
          f"pos={int(cohort[LABEL].sum())}")

    # Render any missing test images (test ECGs aren't precomputed during training).
    # save_all_images skips fileIDs whose PNG already exists, so this is cheap on reruns.
    print("Precomputing test images (skips existing)...")
    save_all_images(cohort, IMAGE_DIR)

    print("Loading test images...")
    images = load_images_parallel(cohort['fileID'], IMAGE_DIR)
    cohort['image_array'] = cohort['fileID'].map(images)
    n_before = len(cohort)
    cohort = cohort[cohort['image_array'].notna()].reset_index(drop=True)
    print(f"  loaded {len(cohort)}/{n_before} images ({n_before - len(cohort)} failed to render)")
    assert len(cohort) > 0, f"No images loaded from {IMAGE_DIR}"
    X = np.stack(cohort['image_array'].values)

    # --- evaluate each model on the same X ---
    rows = []
    for path in model_paths:
        name = os.path.basename(path)
        print(f"\n=== {name} ===")
        model = tf.keras.models.load_model(path, custom_objects={'loss_fn': loss_fn})
        c = cohort.copy()
        c['pred'] = predict(model, X)
        pat = patient_level(c)

        y, s = pat['label'].values, pat['pred_mean'].values
        auroc = roc_auc_score(y, s)
        auprc = average_precision_score(y, s)
        sens, _ = sensitivity_at_specificity(y, s, 0.90)
        spec, _ = specificity_at_sensitivity(y, s, 0.90)
        print(f"  patient-level AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
              f"Sens@90spec={sens:.4f}  Spec@90sens={spec:.4f}")
        rows.append({'model': name, 'n_patients': len(pat), 'pos': int(y.sum()),
                     'auroc': auroc, 'auprc': auprc,
                     'sens_90spec': sens, 'spec_90sens': spec})
        del model
        tf.keras.backend.clear_session()

    # --- comparison table ---
    df = pd.DataFrame(rows).sort_values('auroc', ascending=False)
    print("\n" + "=" * 70)
    print("TEST-SET COMPARISON (patient-level, mean score per MRN)")
    print("=" * 70)
    print(df.to_string(index=False))
    if len(df) > 1:
        best = df.iloc[0]
        print(f"\nBest: {best['model']}  (AUROC={best['auroc']:.4f})")

    save_date = datetime.today().strftime('%Y_%m_%d')
    out = os.path.join(project.tabs_path, f'test_comparison_{save_date}.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
