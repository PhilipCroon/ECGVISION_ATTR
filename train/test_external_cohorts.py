"""
Score OLD vs NEW on the external image cohorts (WVU, Northwell, ...). These folders
already contain ECG images, so no rendering — just the deployment make_image
preprocessing (L->RGB, skimage resize 300, [0,1]) + predict. Per-image AUROC per
cohort (case folder = 1, control folder = 0).

Paths mirror multimodal_amyloid run_ext_val/run_models_ext.py. Run:
    python train/test_external_cohorts.py
"""
import os
import re
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
from skimage.transform import resize as sk_resize
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project
from model import loss_fn
from eval import sensitivity_at_specificity, specificity_at_sensitivity
from eval_log import log_eval

MODEL_DIR = os.path.join(project.project_root, 'models')
_nm = os.getenv('NEW_MODEL', 'attr_amyloid_2026_06_05_unfrozen_06')
NEW_MODEL = _nm if os.path.isabs(_nm) else os.path.join(MODEL_DIR, _nm)
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

# Greece is PDF-based: one ECG PDF per file in a single folder + a labels CSV
# (no pre-split case/control dirs). Label joined via the 'ECG_<id>' filename marker.
# RAW path — NO YOLO crop (deploy crops; we skip it). make_image dispatches on the
# .pdf suffix: convert page0 @ dpi=300 -> same L->RGB->resize300 as the image path.
GREECE = {
    'pdf_dir':    '/mnt/nfs_yale_ecg/amyloid/Greece_Amyloid/Greece_PDFs2',
    'labels_csv': '/mnt/nfs_yale_ecg/amyloid/Greece_Dems.csv',
}


# ---- Greece patient-id + label helpers (ported from multimodal_amyloid run_models_ext.py) ----
def _normalize_patient_id(value):
    if value is None:
        return ""
    cleaned = str(value).strip().lstrip("﻿")
    if not cleaned:
        return ""
    cleaned = cleaned.translate({
        ord("Ε"): "E", ord("Χ"): "X",   # Greek capital epsilon / chi
        ord("ε"): "E", ord("χ"): "X",   # Greek small epsilon / chi
    })
    cleaned = re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
    if not cleaned:
        return ""
    if cleaned.isdigit():
        return cleaned.lstrip("0") or "0"
    if cleaned.replace(".", "", 1).isdigit() and cleaned.count(".") == 1:
        try:
            return str(int(float(cleaned)))
        except ValueError:
            return cleaned
    return cleaned


def _normalize_header(value):
    return " ".join(value.lstrip("﻿").strip().lower().replace("_", " ").replace("-", " ").split())


def _greece_frame(cfg):
    """Build a (cohort,label,path) frame for Greece by matching PDFs to the labels CSV."""
    labels_df = pd.read_csv(cfg['labels_csv'], encoding='utf-8-sig')
    hdr = {_normalize_header(c): c for c in labels_df.columns}
    pid_col, lab_col = hdr.get("patient id"), hdr.get("label")
    assert pid_col and lab_col, \
        f"Need 'Patient ID'+'Label' in {cfg['labels_csv']}, got {list(labels_df.columns)}"
    label_map = {}
    for _, row in labels_df.iterrows():
        pid = _normalize_patient_id(row.get(pid_col))
        if pid:
            label_map[pid] = 1 if str(row.get(lab_col) or "").strip().upper() == "ATTR" else 0

    pdfs = sorted(os.path.join(r, f)
                  for r, _, files in os.walk(cfg['pdf_dir'])
                  for f in files if f.lower().endswith('.pdf'))
    recs = []
    for p in pdfs:
        stem = os.path.splitext(os.path.basename(p))[0]
        idx = stem.upper().find("ECG_")                      # id follows the 'ECG_' marker
        pid = _normalize_patient_id(stem[idx + 4:]) if idx != -1 else ""
        recs.append({'cohort': 'Greece', 'label': label_map.get(pid), 'path': p})
    df = pd.DataFrame(recs)
    matched = df['label'].notna().sum()
    print(f"Greece    {len(pdfs)} PDFs, matched {matched} to labels "
          f"({len(df) - matched} unmatched -> dropped)  ({cfg['pdf_dir']})")
    return df[df['label'].notna()].assign(label=lambda d: d['label'].astype(int))


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
    # exact deployment preprocessing (ynhh-apis app.make_image). PDFs (Greece) are
    # rasterized first: page0 @ dpi=300, same as deploy (data_api main.py:108-114),
    # then the identical L->RGB->resize300. NO YOLO crop.
    try:
        if path.lower().endswith('.pdf'):
            pages = convert_from_path(path, dpi=300, first_page=1, last_page=1)
            if not pages:
                return None
            arr = np.array(pages[0].convert('RGB'))
        else:
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
    if os.path.isdir(GREECE['pdf_dir']):
        frames.append(_greece_frame(GREECE))
    else:
        print(f"Greece    skipped — PDF dir not found ({GREECE['pdf_dir']})")
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

    # Eyeball backstop for the unvalidated PDF path: dump a few Greece renders.
    # If the ECG isn't a tight 12-lead plot filling the 300x300 frame, the no-crop
    # preprocessing is off and Greece AUROCs are meaningless — check before trusting.
    g = df[df['cohort'] == 'Greece']
    if len(g):
        check_dir = os.path.join(project.tabs_path, 'greece_render_check')
        os.makedirs(check_dir, exist_ok=True)
        for i in range(min(5, len(g))):
            Image.fromarray((g['img'].iloc[i] * 255).astype(np.uint8)).save(
                os.path.join(check_dir, f"sample_{i}_label{g['label'].iloc[i]}.png"))
        print(f"Saved {min(5, len(g))} Greece render samples -> {check_dir} (EYEBALL THESE)")

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    rows = []
    pred_tags = []
    for tag, path in [('NEW', NEW_MODEL), ('OLD', OLD_MODEL)]:
        if not os.path.exists(path):
            print(f"{tag} MISSING: {path}")
            continue
        print(f"\n=== {tag}: {os.path.basename(path)} ===")
        model = tf.keras.models.load_model(path, custom_objects={'loss_fn': loss_fn})
        df[f'pred_{tag}'] = predict(model, X)
        pred_tags.append(tag)
        for name, g in df.groupby('cohort'):
            y, s = g['label'].values, g[f'pred_{tag}'].values
            if len(np.unique(y)) < 2:
                print(f"  {name}: only one class, skipping AUROC")
                continue
            auroc = roc_auc_score(y, s)
            auprc = average_precision_score(y, s)
            sens, _ = sensitivity_at_specificity(y, s, 0.90)
            spec, _ = specificity_at_sensitivity(y, s, 0.90)
            print(f"  {name:10s} n={len(g):4d} pos={int(y.sum()):4d} "
                  f"AUROC={auroc:.4f} AUPRC={auprc:.4f} Sens@90spec={sens:.4f} Spec@90sens={spec:.4f}")
            m = {'n': len(g), 'pos': int(y.sum()), 'auroc': auroc, 'auprc': auprc,
                 'sens_90spec': sens, 'spec_90sens': spec}
            rows.append({'model': tag, 'cohort': name, **m})
            log_eval(path, cohort=name, metrics=m, render='deploy_image', level='image')
        del model
        tf.keras.backend.clear_session()

    out = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("EXTERNAL COHORTS (per-image AUROC)")
    print("=" * 78)
    print(out.to_string(index=False))
    p = os.path.join(project.tabs_path, 'external_cohort_comparison.csv')
    out.to_csv(p, index=False)

    # Per-sample predictions for bootstrap CIs (bootstrap_ci.py).
    keep = ['cohort', 'label', 'path'] + [f'pred_{t}' for t in pred_tags]
    preds_path = os.path.join(project.tabs_path, 'external_preds.csv')
    df[keep].to_csv(preds_path, index=False)
    print(f"Saved per-sample predictions: {preds_path}")
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
