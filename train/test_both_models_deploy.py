"""
Compare OLD vs NEW on the test set, rendered with the DEPLOYMENT (ynhh-apis) code
instead of make_plot — so the test images match what the amyloid API actually feeds
the model in production (raw DICOM -> ECG.signals (12,5000) mV, Butterworth 40 Hz ->
process_ecg_plot_from_signal (bw_alt, 4-col, dpi 100) -> make_image (L->RGB,
skimage.resize 300x300 -> [0,1])).

Train still uses make_plot's random-style augmentation (unchanged). Only the TEST
render switches to the deployment pipeline, per the deployment-faithful eval plan.

Requires the ynhh-apis repo on the box. Set YNHH_APIS_PATH if not at the default:
    git clone git@github.com:CarDS-Yale/ynhh-apis.git ~/projects/ynhh-apis
    python train/test_both_models_deploy.py
"""
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize as sk_resize
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project
from model import loss_fn
from eval import (load_test_cohort, sensitivity_at_specificity,
                  specificity_at_sensitivity)

sys.path.append(project.ecg_image_models_path)
from utils import load_signal_mV

# --- ynhh-apis deployment render code (reused verbatim) ---
YNHH = os.getenv('YNHH_APIS_PATH', '/home/pmc57/projects/ynhh-apis/amyloid-api')
sys.path.append(YNHH)
from ecg.ecg import process_ecg_plot_from_signal, butter_lowpass_filter  # noqa: E402

LABEL = 'amyloid'
MODEL_DIR = os.path.join(project.project_root, 'models')
NEW_MODEL = os.path.join(MODEL_DIR, 'attr_amyloid_2026_06_05_unfrozen_06')
OLD_MODEL = '/home/pmc57/cmp-jdat-data/variant_amyloid/models/trained_model_Amyloidosis_stage2_age_sex_1_10_15'
IMAGE_DIR = project.image_dir + '_test_deploy'


def render_deploy(fid):
    """Render one ECG with the deployment pipeline; return the PNG path (or None)."""
    out = os.path.join(IMAGE_DIR, os.path.basename(str(fid)) + '.png')
    if os.path.exists(out):
        return out
    try:
        sig = load_signal_mV(fid).T.astype(np.float32)          # (12, 5000) mV
        if sig.shape != (12, 5000):
            return None
        # match ECG._signals: Butterworth 40 Hz lowpass, order 2, per lead @500 Hz
        for i in range(12):
            sig[i] = butter_lowpass_filter(sig[i], 40.0, 500.0, 2)
        process_ecg_plot_from_signal(sig, os.path.basename(str(fid)), out)
        return out if os.path.exists(out) else None
    except Exception as e:
        print(f"render fail {fid}: {e}")
        return None


def make_image(path):
    # exact deployment preprocessing (ynhh-apis app.make_image)
    arr = np.array(Image.open(path))
    return sk_resize(np.array(Image.fromarray(arr).convert('L').convert('RGB')), (300, 300))


def predict(model, X, batch_size=256):
    preds = []
    for i in tqdm(range(int(np.ceil(len(X) / batch_size))), desc="  predicting", leave=False):
        preds.append(model.predict_on_batch(X[i*batch_size:(i+1)*batch_size]))
    return np.concatenate(preds).squeeze()


def patient_level(c):
    return (c.groupby('MRN').agg(pred_mean=('pred', 'mean'),
                                 label=(LABEL, 'max'), group=('group', 'first'))
            .reset_index())


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    models = [('NEW', NEW_MODEL), ('OLD', OLD_MODEL)]
    print(f"Deployment-style render dir: {IMAGE_DIR}")
    for tag, p in models:
        print(f"  {tag} [{'ok' if os.path.exists(p) else 'MISSING'}]: {p}")
    assert os.path.exists(NEW_MODEL), f"NEW missing: {NEW_MODEL}"
    models = [(t, p) for t, p in models if os.path.exists(p)]

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    cohort = load_test_cohort()
    print(f"\nTest ECGs: {len(cohort)} ({cohort['MRN'].nunique()} MRNs)  pos={int(cohort[LABEL].sum())}")

    print("Rendering with deployment pipeline (skips existing)...")
    paths = [render_deploy(f) for f in tqdm(cohort['fileID'], desc="render")]
    cohort['img_path'] = paths
    n0 = len(cohort)
    cohort = cohort[cohort['img_path'].notna()].reset_index(drop=True)
    print(f"  rendered {len(cohort)}/{n0} ({n0 - len(cohort)} failed)")
    assert len(cohort) > 0, "0 deployment renders"

    print("Loading images (deployment make_image: L->RGB, skimage resize, [0,1])...")
    X = np.stack([make_image(p) for p in tqdm(cohort['img_path'], desc="load")]).astype(np.float32)

    rows = []
    for tag, path in models:
        print(f"\n=== {tag}: {os.path.basename(path)} ===")
        model = tf.keras.models.load_model(path, custom_objects={'loss_fn': loss_fn})
        c = cohort.copy()
        c['pred'] = predict(model, X)
        pat = patient_level(c)
        y, s = pat['label'].values, pat['pred_mean'].values
        auroc, auprc = roc_auc_score(y, s), average_precision_score(y, s)
        sens, _ = sensitivity_at_specificity(y, s, 0.90)
        spec, _ = specificity_at_sensitivity(y, s, 0.90)
        print(f"  AUROC={auroc:.4f} AUPRC={auprc:.4f} Sens@90spec={sens:.4f} Spec@90sens={spec:.4f}")
        rows.append({'tag': tag, 'model': os.path.basename(path), 'n_patients': len(pat),
                     'pos': int(y.sum()), 'auroc': auroc, 'auprc': auprc,
                     'sens_90spec': sens, 'spec_90sens': spec})
        del model
        tf.keras.backend.clear_session()

    df = pd.DataFrame(rows).sort_values('auroc', ascending=False)
    print("\n" + "=" * 78)
    print("TEST COMPARISON — DEPLOYMENT RENDER (patient-level, mean per MRN)")
    print("=" * 78)
    print(df.to_string(index=False))
    out = os.path.join(project.tabs_path, 'test_comparison_deploy.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
