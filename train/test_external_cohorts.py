"""
Score OLD vs NEW on the external image cohorts (WVU, Northwell, ...). These folders
already contain ECG images, so no rendering — just the deployment make_image
preprocessing (L->RGB, skimage resize 300, [0,1]) + predict. Per-image AUROC per
cohort (case folder = 1, control folder = 0).

Paths mirror multimodal_amyloid run_ext_val/run_models_ext.py. Run:
    python train/test_external_cohorts.py
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
from eval import sensitivity_at_specificity, specificity_at_sensitivity

MODEL_DIR = os.path.join(project.project_root, 'models')
NEW_MODEL = os.path.join(MODEL_DIR, 'attr_amyloid_2026_06_05_unfrozen_06')
OLD_MODEL = '/home/pmc57/cmp-jdat-data/variant_amyloid/models/trained_model_Amyloidosis_stage2_age_sex_1_10_15'

IMG_SUFFIXES = ('.png', '.jpg', '.jpeg')

# case folder -> label 1, control folder -> label 0 (paths from run_models_ext.py)
COHORTS = {
    'WVU': {
        'case':    '/mnt/nfs_yale_ecg/amyloid/WVU_Amyloid',
        'control': '/mnt/nfs_yale_ecg/amyloid/WVU_ECG_controls',
    },
    'Northwell': {
        'case':    '/mnt/nfs_yale_ecg/amyloid/Northwell_Amyloid',
        'control': '/mnt/nfs_yale_ecg/amyloid/Northwell_Controls',
    },
}


def list_images(folder):
    out = []
    if not os.path.isdir(folder):
        return out
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_SUFFIXES):
                out.append(os.path.join(root, f))
    return sorted(out)


def make_image(path):
    # exact deployment preprocessing (ynhh-apis app.make_image)
    try:
        arr = np.array(Image.open(path))
        return sk_resize(np.array(Image.fromarray(arr).convert('L').convert('RGB')), (300, 300)).astype(np.float32)
    except Exception as e:
        print(f"load fail {path}: {e}")
        return None


def predict(model, X, batch_size=256):
    preds = []
    for i in tqdm(range(int(np.ceil(len(X) / batch_size))), desc="  predicting", leave=False):
        preds.append(model.predict_on_batch(X[i*batch_size:(i+1)*batch_size]))
    return np.concatenate(preds).squeeze()


def main():
    # --- gather files + labels per cohort ---
    frames = []
    for name, folders in COHORTS.items():
        for lab_name, label in (('case', 1), ('control', 0)):
            files = list_images(folders[lab_name])
            print(f"{name:10s} {lab_name:8s} {len(files):5d} images  ({folders[lab_name]})")
            frames.append(pd.DataFrame({'cohort': name, 'label': label, 'path': files}))
    df = pd.concat(frames, ignore_index=True)
    if len(df) == 0:
        print("No images found — check the cohort paths.")
        return

    # --- load images in parallel, BEFORE GPU init (CUDA not fork-safe) ---
    nworkers = max(1, cpu_count() - 2)
    print(f"\nLoading {len(df)} images with {nworkers} workers...")
    with Pool(nworkers) as pool:
        imgs = list(tqdm(pool.imap(make_image, df['path'].tolist(), chunksize=8),
                         total=len(df), desc="load"))
    df['img'] = imgs
    df = df[df['img'].notna()].reset_index(drop=True)
    X = np.stack(df['img'].values).astype(np.float32)
    print(f"loaded {len(df)} images")

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    rows = []
    for tag, path in [('NEW', NEW_MODEL), ('OLD', OLD_MODEL)]:
        if not os.path.exists(path):
            print(f"{tag} MISSING: {path}")
            continue
        print(f"\n=== {tag}: {os.path.basename(path)} ===")
        model = tf.keras.models.load_model(path, custom_objects={'loss_fn': loss_fn})
        df['pred'] = predict(model, X)
        for name, g in df.groupby('cohort'):
            y, s = g['label'].values, g['pred'].values
            if len(np.unique(y)) < 2:
                print(f"  {name}: only one class, skipping AUROC")
                continue
            auroc = roc_auc_score(y, s)
            auprc = average_precision_score(y, s)
            sens, _ = sensitivity_at_specificity(y, s, 0.90)
            spec, _ = specificity_at_sensitivity(y, s, 0.90)
            print(f"  {name:10s} n={len(g):4d} pos={int(y.sum()):4d} "
                  f"AUROC={auroc:.4f} AUPRC={auprc:.4f} Sens@90spec={sens:.4f} Spec@90sens={spec:.4f}")
            rows.append({'model': tag, 'cohort': name, 'n': len(g), 'pos': int(y.sum()),
                         'auroc': auroc, 'auprc': auprc, 'sens_90spec': sens, 'spec_90sens': spec})
        del model
        tf.keras.backend.clear_session()

    out = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("EXTERNAL COHORTS (per-image AUROC)")
    print("=" * 78)
    print(out.to_string(index=False))
    p = os.path.join(project.tabs_path, 'external_cohort_comparison.csv')
    out.to_csv(p, index=False)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
