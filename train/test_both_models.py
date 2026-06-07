"""
Compare the NEW vs OLD amyloid model on the test set — one self-contained script.

    python test_both_models.py

Both checkpoint paths are baked in. Test ECGs are rendered ONCE with a FIXED
standard 12-lead layout (deterministic=True: 'alternate' 2-col, bw, fixed
font/line) into a dedicated *_test_std dir, so AUROC reflects the model, not a
lucky/unlucky render. Both models score on the identical clean images.

Test set = tabs/cohort_test.csv (never seen during training). Metrics are
patient-level (mean score per MRN).
"""
import os
import sys
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
from eval import (load_test_cohort, sensitivity_at_specificity,
                  specificity_at_sensitivity)

sys.path.append(project.ecg_image_models_path)
from utils import load_images_parallel, save_all_images

LABEL = 'amyloid'
MODEL_DIR = os.path.join(project.project_root, 'models')

# --- the two models to compare (baked in) ---
NEW_MODEL = os.path.join(MODEL_DIR, 'attr_amyloid_2026_06_05_unfrozen_06')
OLD_MODEL = '/home/pmc57/projects/multimodal_amyloid/models/trained_model_Amyloidosis_stage2_age_sex_1_10_15'

# Dedicated dir for deterministic standard-layout renders — kept separate from the
# random-style training PNGs so caching works and there's no collision.
IMAGE_DIR = project.image_dir + '_test_std'


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
    models = [('NEW', NEW_MODEL), ('OLD', OLD_MODEL)]
    print("Comparing on test set (standard-layout render):")
    for tag, p in models:
        ok = "ok" if os.path.exists(p) else "MISSING"
        print(f"  {tag} [{ok}]: {p}")
    assert os.path.exists(NEW_MODEL), f"NEW model not found at {NEW_MODEL}"
    models = [(t, p) for t, p in models if os.path.exists(p)]
    if len(models) < 2:
        print("WARNING: an expected model is missing — continuing with what exists.")

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # --- load test cohort + render std images ONCE ---
    cohort = load_test_cohort()
    print(f"\nTest ECGs: {len(cohort)} ({cohort['MRN'].nunique()} MRNs)  "
          f"pos={int(cohort[LABEL].sum())}")

    print("Precomputing standard-layout test images (skips existing)...")
    status = save_all_images(cohort, IMAGE_DIR, deterministic=True)
    print(f"  render status: {status}")
    if status.get('saved', 0) == 0 and status.get('skipped', 0) == 0:
        raise RuntimeError(
            f"save_all_images rendered 0 PNGs (status={status}). Signals likely not "
            f"found — check {IMAGE_DIR} and the make_plot signal search paths.")

    print("Loading test images...")
    images = load_images_parallel(cohort['fileID'], IMAGE_DIR)
    cohort['image_array'] = cohort['fileID'].map(images)
    n_before = len(cohort)
    cohort = cohort[cohort['image_array'].notna()].reset_index(drop=True)
    print(f"  loaded {len(cohort)}/{n_before} images "
          f"({n_before - len(cohort)} failed to render)")
    assert len(cohort) > 0, f"No images loaded from {IMAGE_DIR}"
    X = np.stack(cohort['image_array'].values)

    # --- evaluate each model on the same X ---
    rows = []
    for tag, path in models:
        name = os.path.basename(path)
        print(f"\n=== {tag}: {name} ===")
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
        rows.append({'tag': tag, 'model': name, 'n_patients': len(pat),
                     'pos': int(y.sum()), 'auroc': auroc, 'auprc': auprc,
                     'sens_90spec': sens, 'spec_90sens': spec})
        del model
        tf.keras.backend.clear_session()

    # --- comparison table ---
    df = pd.DataFrame(rows).sort_values('auroc', ascending=False)
    print("\n" + "=" * 78)
    print("TEST-SET COMPARISON (patient-level, mean score per MRN, standard layout)")
    print("=" * 78)
    print(df.to_string(index=False))
    if len(df) > 1:
        best = df.iloc[0]
        print(f"\nBest: {best['tag']} ({best['model']})  AUROC={best['auroc']:.4f}")

    save_date = datetime.today().strftime('%Y_%m_%d')
    out = os.path.join(project.tabs_path, f'test_comparison_{save_date}.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
