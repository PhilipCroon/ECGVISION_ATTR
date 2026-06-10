"""
Pre-crop ECG PDFs -> cropped PNGs using the deploy YOLO cropper.

Torch/ultralytics ONLY — NO tensorflow imported here, so this runs in a SEPARATE
env from scoring (the box's torch CUDA build conflicts with the TF env's CUDA libs).
Pipeline matches deploy (ynhh-apis data_api/app/model.py YOLOCropper):
  raster PDF page0 @ dpi -> YOLO detect ECG box -> crop largest -> rotate -90 if portrait.

Output PNGs keep the SOURCE FILENAME STEM (so downstream StudyID/label parsing in
test_external_cohorts.py still works). Then score with CROP off, pointing the cohort
at this output dir (SCAN_MP['img_dir'] auto-detects it).

Multiprocessed: a worker Pool (CPU torch -> no CUDA fork issue). Each worker loads its
OWN YOLO model once (Pool initializer) and is pinned to 1 torch thread to avoid
core oversubscription. GPU (--device 0) forces workers=1 (CUDA not fork-safe).

Usage (in a torch-only env):
  python crop_pdfs.py \
    --pdf-dir /home/pmc57/projects/multimodal_amyloid/scan_mp/scanmp_ecg_pdf_cleaned \
    --out-dir /home/pmc57/projects/multimodal_amyloid/scan_mp/scanmp_ecg_png_cropped \
    --weights /home/pmc57/projects/ynhh-apis/AI_ECG_server/data_api/models/train5_yolo/weights/best.pt \
    --workers 16
"""
import argparse
import os
from multiprocessing import Pool, cpu_count

from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

# per-worker globals (set in _init_worker)
_MODEL = None
_CONF = 0.8
_DPI = 300
_DEVICE = 'cpu'
_OUT_DIR = None
_OVERWRITE = False


def _init_worker(weights, conf, dpi, device, out_dir, overwrite, threads):
    global _MODEL, _CONF, _DPI, _DEVICE, _OUT_DIR, _OVERWRITE
    import torch
    torch.set_num_threads(max(1, threads))   # avoid oversubscription across workers
    from ultralytics import YOLO
    _MODEL = YOLO(weights)
    _CONF, _DPI, _DEVICE, _OUT_DIR, _OVERWRITE = conf, dpi, device, out_dir, overwrite


def _crop_pil(pil):
    results = _MODEL.predict(pil, conf=_CONF, device=_DEVICE, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return pil, False
    boxes = results[0].boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    cropped = pil.crop((int(x1), int(y1), int(x2), int(y2)))
    if cropped.height > cropped.width:
        cropped = cropped.rotate(-90, expand=True)
    return cropped, True


def _process(pdf_path):
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(_OUT_DIR, f"{stem}.png")
    if os.path.exists(out_path) and not _OVERWRITE:
        return 'skip'
    try:
        pages = convert_from_path(pdf_path, dpi=_DPI, first_page=1, last_page=1)
        if not pages:
            return 'fail'
        cropped, did = _crop_pil(pages[0].convert('RGB'))
        cropped.save(out_path, format='PNG')
        return 'cropped' if did else 'full'
    except Exception as e:
        print(f"fail {stem}: {e}")
        return 'fail'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--conf', type=float, default=0.8)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--device', default='cpu', help="'cpu' (default) or a GPU index like '0'")
    ap.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    assert os.path.exists(args.weights), f"YOLO weights not found: {args.weights}"
    os.makedirs(args.out_dir, exist_ok=True)

    pdfs = sorted(os.path.join(r, f)
                  for r, _, files in os.walk(args.pdf_dir)
                  for f in files if f.lower().endswith('.pdf'))
    assert pdfs, f"No PDFs in {args.pdf_dir}"

    workers = args.workers
    if args.device != 'cpu':
        workers = 1   # CUDA not fork-safe -> single process on GPU
        print("GPU device set -> forcing workers=1 (CUDA not fork-safe)")
    workers = max(1, min(workers, len(pdfs)))
    # split CPU threads across workers so torch doesn't oversubscribe cores
    threads = max(1, (cpu_count() - 1) // workers)
    print(f"{len(pdfs)} PDFs -> {args.out_dir} "
          f"(conf={args.conf}, dpi={args.dpi}, device={args.device}, "
          f"workers={workers}, threads/worker={threads})")

    init_args = (args.weights, args.conf, args.dpi, args.device,
                 args.out_dir, args.overwrite, threads)
    counts = {'cropped': 0, 'full': 0, 'skip': 0, 'fail': 0}
    with Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
        for status in tqdm(pool.imap_unordered(_process, pdfs, chunksize=4),
                           total=len(pdfs), desc="crop"):
            counts[status] += 1

    n_ok = counts['cropped'] + counts['full']
    print(f"\nwrote {n_ok} PNGs ({counts['cropped']} YOLO-cropped, "
          f"{counts['full']} no-detection/full-frame), "
          f"{counts['skip']} skipped, {counts['fail']} failed")
    print(f"Now score with CROP off, pointing the cohort img_dir at: {args.out_dir}")


if __name__ == '__main__':
    main()
