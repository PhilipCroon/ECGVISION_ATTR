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
from PIL import Image, ImageEnhance
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

# Optional YOLO ECG-region crop, matching the deploy pipeline (ynhh-apis
# data_api/app/model.py YOLOCropper). OFF by default to preserve the no-crop deltas.
# CROP=1 enables it for ALL cohorts (deploy fidelity). Scanned printouts (SCAN-MP)
# have the ECG small in a full page -> no-crop resize makes it tiny/OOD; crop fixes that.
CROP_ENABLED = os.getenv('CROP', '0') == '1'
YOLO_WEIGHTS = os.getenv('YOLO_WEIGHTS', '')
YOLO_CONF = float(os.getenv('YOLO_CONF', '0.8'))

# Optional brightness/contrast normalization for scanned/real-world ECGs, ported from
# upstream ecg-image-models (fix_brightness_and_contrast, used in DataSequenceTest_Adjust).
# Scanned ECGs (e.g. SCAN-MP) are darker/lower-contrast than the clean training renders;
# this nudges the central region to the training band (mean~0.88, std~0.10) so the model
# can read them. ADJUST=1 applies it to ALL cohorts before resize. No retraining.
ADJUST_ENABLED = os.getenv('ADJUST', '0') == '1'

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

# SCAN-MP: same PDF/no-crop pattern as Greece, different label source. Labels in an
# xlsx keyed by 'study ID'; PDFs named 'id-<studyid>_...'. PyP status -> binary via
# PYP_STATUS_MAP (ported from multimodal_amyloid SCAN_MP_sensitivity_analysis.py);
# statuses not in the map -> dropped.
SCAN_MP = {
    'pdf_dir':     '/home/pmc57/projects/multimodal_amyloid/scan_mp/scanmp_ecg_pdf_cleaned',
    # Pre-cropped PNGs from crop_pdfs.py (torch-only env). If this dir exists it is
    # used INSTEAD of pdf_dir -> score with CROP off (already cropped). Same filename
    # stems, so the id-<studyid> label join is unchanged.
    'img_dir':     '/home/pmc57/projects/multimodal_amyloid/scan_mp/scanmp_ecg_png_cropped',
    'labels_xlsx': '/home/pmc57/projects/multimodal_amyloid/scan_mp/SCAN-MP_labels.xlsx',
}
PYP_STATUS_MAP = {
    "Strongly suggestive of ATTR": 1,
    "Equivocal for ATTR / Possible early ATTR": 0,
    "ATTR by EMB": 1,
    "Not suggestive of ATTR": 0,
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


def _scanmp_frame(cfg):
    """Build a (cohort,label,path) frame for SCAN-MP. PDFs named 'id-<studyid>_...';
    label from the xlsx 'study ID' -> 'Amyloid_status_by_PYP' via PYP_STATUS_MAP."""
    lab = pd.read_excel(cfg['labels_xlsx']).rename(
        columns={"study ID": "StudyID", "Amyloid_status_by_PYP": "PyP_status"})
    lab["Amyloid"] = lab["PyP_status"].map(PYP_STATUS_MAP)
    lab = lab.dropna(subset=["Amyloid"])
    label_map = {int(sid): int(a) for sid, a in zip(lab["StudyID"], lab["Amyloid"])}

    def studyid(stem):
        try:
            return int(stem.split("_")[0].replace("id-", ""))
        except (ValueError, IndexError):
            return None

    # Prefer pre-cropped PNGs (crop_pdfs.py output) if that dir exists; else raw PDFs.
    img_dir = cfg.get('img_dir')
    if img_dir and os.path.isdir(img_dir):
        src_dir, sfx, kind = img_dir, IMG_SUFFIXES, 'pre-cropped PNGs'
    else:
        src_dir, sfx, kind = cfg['pdf_dir'], ('.pdf',), 'PDFs (no crop)'
    files = sorted(os.path.join(r, f)
                   for r, _, fs in os.walk(src_dir)
                   for f in fs if f.lower().endswith(sfx))
    recs = []
    for p in files:
        sid = studyid(os.path.splitext(os.path.basename(p))[0])
        recs.append({'cohort': 'SCAN_MP', 'label': label_map.get(sid), 'path': p})
    df = pd.DataFrame(recs)
    matched = df['label'].notna().sum()
    print(f"SCAN_MP   {len(files)} {kind}, matched {matched} to labels "
          f"({len(df) - matched} unmatched -> dropped)  ({src_dir})")
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


def _raster_pil(path):
    # Load a file to a full-res RGB PIL image. PDFs (Greece/SCAN-MP) -> page0 @ dpi=300
    # (same as deploy data_api main.py:108-114). No resize/grayscale yet.
    if path.lower().endswith('.pdf'):
        pages = convert_from_path(path, dpi=300, first_page=1, last_page=1)
        if not pages:
            return None
        return pages[0].convert('RGB')
    return Image.fromarray(np.array(Image.open(path))).convert('RGB')


def fix_brightness_and_contrast(img):
    # Ported verbatim from upstream ecg-image-models utils_hcm.py. Iteratively enhances
    # a grayscale image so the central [50:250,50:250] region's mean brightness lands in
    # [0.87,0.89] and contrast (std) in [0.09,0.11], matching the training distribution.
    pil_l = Image.fromarray(img).convert('L')
    bright = np.mean(sk_resize(np.array(pil_l), (300, 300))[50:250, 50:250])
    counter = 0
    while (bright > .89 or bright < .87) and counter <= 100:
        factor = 1.05 if bright < .87 else 0.95
        pil_l = ImageEnhance.Brightness(pil_l).enhance(factor)
        bright = np.mean(sk_resize(np.array(pil_l), (300, 300))[50:250, 50:250])
        counter += 1
    con = np.std(sk_resize(np.array(pil_l), (300, 300))[50:250, 50:250])
    counter = 0
    while (con > .11 or con < .09) and counter <= 100:
        factor = 1.05 if con < .09 else 0.95
        pil_l = ImageEnhance.Contrast(pil_l).enhance(factor)
        con = np.std(sk_resize(np.array(pil_l), (300, 300))[50:250, 50:250])
        counter += 1
    return np.array(pil_l.convert('RGB'))


def _finish(pil):
    # amyloid-service preprocessing: L->RGB->resize300, [0,1] float32.
    # ADJUST: normalize brightness/contrast first (scanned/real-world ECGs).
    if ADJUST_ENABLED:
        arr = fix_brightness_and_contrast(np.array(pil.convert('RGB')))
    else:
        arr = np.array(pil.convert('L').convert('RGB'))
    return sk_resize(arr, (300, 300)).astype(np.float32)


def make_image(path):
    # No-crop path (runs in the fork Pool): raster -> finish. Deploy preprocessing
    # minus the YOLO crop.
    try:
        pil = _raster_pil(path)
        return _finish(pil) if pil is not None else None
    except Exception as e:
        print(f"load fail {path}: {e}")
        return None


class YOLOCropper:
    """Ported from ynhh-apis data_api/app/model.py: detect ECG box, crop largest,
    rotate -90 if portrait. Returns (pil, cropped?). NEVER run inside the fork Pool
    (torch+CUDA not fork-safe) — instantiate + call in the main process."""
    def __init__(self, weights_path):
        from ultralytics import YOLO   # imported lazily so non-crop runs don't need it
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
        self.model = YOLO(weights_path)

    def crop(self, pil, conf):
        results = self.model.predict(pil, conf=conf, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return pil, False
        boxes = results[0].boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cropped = pil.crop((int(x1), int(y1), int(x2), int(y2)))
        if cropped.height > cropped.width:
            cropped = cropped.rotate(-90, expand=True)
        return cropped, True


def load_raw_then_crop(path, cropper, conf):
    # CROP path (sequential, main process): raster full-res -> YOLO crop -> finish.
    # One full-res image in RAM at a time (no holding 645 @ 300dpi).
    try:
        pil = _raster_pil(path)
        if pil is None:
            return None, False
        cropped, did = cropper.crop(pil, conf)
        return _finish(cropped), did
    except Exception as e:
        print(f"load fail {path}: {e}")
        return None, False


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
    if os.path.isdir(SCAN_MP['pdf_dir']) or os.path.isdir(SCAN_MP.get('img_dir', '')):
        frames.append(_scanmp_frame(SCAN_MP))
    else:
        print(f"SCAN_MP   skipped — no PDF/img dir ({SCAN_MP['pdf_dir']})")
    df = pd.concat(frames, ignore_index=True)
    if len(df) == 0:
        print("No images found — check the cohort paths.")
        return

    if CROP_ENABLED:
        # YOLO crop path: sequential in main process (torch+CUDA not fork-safe).
        # One full-res image in RAM at a time. Applies to ALL cohorts for deploy fidelity.
        if not YOLO_WEIGHTS:
            raise SystemExit("CROP=1 but YOLO_WEIGHTS unset — point it at the deploy "
                             "best.pt (ynhh-apis .../data_api/models/train4_yolo/weights/best.pt)")
        print(f"\nCROP=1: YOLO crop enabled (weights={YOLO_WEIGHTS}, conf={YOLO_CONF})")
        cropper = YOLOCropper(YOLO_WEIGHTS)
        imgs, crops = [], 0
        for p in tqdm(df['path'].tolist(), desc="raster+crop"):
            img, did = load_raw_then_crop(p, cropper, YOLO_CONF)
            imgs.append(img)
            crops += bool(did)
        df['img'] = imgs
        n_ok = sum(i is not None for i in imgs)
        print(f"cropped {crops}/{n_ok} images (rest: no YOLO detection -> used full frame)")
    else:
        # No-crop path: parallel raster+resize in fork Pool, BEFORE GPU init.
        nworkers = max(1, cpu_count() - 2)
        print(f"\nLoading {len(df)} images with {nworkers} workers (no crop)...")
        with Pool(nworkers) as pool:
            imgs = list(tqdm(pool.imap(make_image, df['path'].tolist(), chunksize=8),
                             total=len(df), desc="load"))
        df['img'] = imgs
    df = df[df['img'].notna()].reset_index(drop=True)
    X = np.stack(df['img'].values).astype(np.float32)
    print(f"loaded {len(df)} images")

    # Eyeball backstop for the unvalidated no-crop PDF path (Greece, SCAN_MP): dump a
    # few renders per cohort. If the ECG isn't a tight 12-lead plot filling the 300x300
    # frame, the no-crop preprocessing is off and those AUROCs are meaningless.
    for cname in ('Greece', 'SCAN_MP'):
        sub = df[df['cohort'] == cname]
        if not len(sub):
            continue
        suffix = ('_crop' if CROP_ENABLED else '') + ('_adjust' if ADJUST_ENABLED else '')
        check_dir = os.path.join(project.tabs_path, f'{cname.lower()}_render_check{suffix}')
        os.makedirs(check_dir, exist_ok=True)
        for i in range(min(5, len(sub))):
            Image.fromarray((sub['img'].iloc[i] * 255).astype(np.uint8)).save(
                os.path.join(check_dir, f"sample_{i}_label{sub['label'].iloc[i]}.png"))
        print(f"Saved {min(5, len(sub))} {cname} render samples -> {check_dir} (EYEBALL THESE)")

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
            log_eval(path, cohort=name, metrics=m,
                     render=('deploy' + ('_yolocrop' if CROP_ENABLED else '_nocrop')
                             + ('_adjust' if ADJUST_ENABLED else '')), level='image')
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
