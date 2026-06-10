"""
Compare OLD vs NEW on the test set, rendered with the DEPLOYMENT (ynhh-apis) code
instead of make_plot — so the test images match what the amyloid API actually feeds
the model in production (raw DICOM -> ECG.signals (12,5000) mV, Butterworth 40 Hz ->
process_ecg_plot_from_signal (bw_alt, 4-col, dpi 100) -> make_image (L->RGB,
skimage.resize 300x300 -> [0,1])).

Train still uses make_plot's random-style augmentation (unchanged). Only the TEST
render switches to the deployment pipeline, per the deployment-faithful eval plan.

Render code is vendored in train/deploy_render.py (copied verbatim from ynhh-apis
amyloid-api/ecg/ecg.py), so no clone is needed. Run:
    python train/test_both_models_deploy.py
"""
import os
import sys
from multiprocessing import Pool, cpu_count

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
from eval_log import log_eval

sys.path.append(project.ecg_image_models_path)
from utils import load_signal_mV

# --- deployment render code (vendored verbatim from ynhh-apis ecg.ecg) ---
from deploy_render import process_ecg_plot_from_signal, butter_lowpass_filter

LABEL = 'amyloid'
MODEL_DIR = os.path.join(project.project_root, 'models')
_nm = os.getenv('NEW_MODEL', 'attr_amyloid_2026_06_05_unfrozen_06')
NEW_MODEL = _nm if os.path.isabs(_nm) else os.path.join(MODEL_DIR, _nm)
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

    cohort = load_test_cohort()
    print(f"\nTest ECGs: {len(cohort)} ({cohort['MRN'].nunique()} MRNs)  pos={int(cohort[LABEL].sum())}")

    # --- render + image load in parallel, BEFORE any TF/GPU init ---
    # CUDA is not fork-safe: fork the pools first, then touch the GPU.
    nworkers = max(1, cpu_count() - 2)
    print(f"Rendering with deployment pipeline ({nworkers} workers, skips existing)...")
    with Pool(nworkers) as pool:
        paths = list(tqdm(pool.imap(render_deploy, cohort['fileID'].tolist(), chunksize=8),
                          total=len(cohort), desc="render"))
    cohort['img_path'] = paths
    n0 = len(cohort)
    cohort = cohort[cohort['img_path'].notna()].reset_index(drop=True)
    print(f"  rendered {len(cohort)}/{n0} ({n0 - len(cohort)} failed)")
    assert len(cohort) > 0, "0 deployment renders"

    print(f"Loading images ({nworkers} workers; deployment make_image: L->RGB, skimage resize, [0,1])...")
    with Pool(nworkers) as pool:
        imgs = list(tqdm(pool.imap(make_image, cohort['img_path'].tolist(), chunksize=8),
                         total=len(cohort), desc="load"))
    X = np.stack(imgs).astype(np.float32)

    # --- now safe to init the GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

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
        log_eval(path, cohort='internal_test', render='deploy', level='patient',
                 metrics={'n': len(pat), 'pos': int(y.sum()), 'auroc': auroc, 'auprc': auprc,
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
