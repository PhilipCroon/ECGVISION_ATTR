"""
Pre-crop ECG PDFs -> cropped PNGs using the deploy YOLO cropper.

Torch/ultralytics ONLY — NO tensorflow imported here, so this runs in a SEPARATE
env from scoring (the box's torch CUDA build conflicts with the TF env's CUDA libs).
Pipeline matches deploy (ynhh-apis data_api/app/model.py YOLOCropper):
  raster PDF page0 @ dpi -> YOLO detect ECG box -> crop largest -> rotate -90 if portrait.

Output PNGs keep the SOURCE FILENAME STEM (so downstream StudyID/label parsing in
test_external_cohorts.py still works). Then score with CROP off, pointing the cohort
at this output dir:
  SCAN_MP['img_dir'] = <out-dir>   (auto-used if it exists; falls back to pdf_dir)

Crop runs on CPU by default (preprocessing, no GPU needed) to dodge CUDA mismatches;
pass --device 0 to use a GPU if torch sees one.

Usage (in a torch-only env):
  python crop_pdfs.py \
    --pdf-dir /home/pmc57/projects/multimodal_amyloid/scan_mp/scanmp_ecg_pdf_cleaned \
    --out-dir /home/pmc57/projects/multimodal_amyloid/scan_mp/scanmp_ecg_png_cropped \
    --weights /path/to/ynhh-apis/AI_ECG_server/data_api/models/train4_yolo/weights/best.pt
"""
import argparse
import os

from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm
from ultralytics import YOLO


def crop_one(model, pil, conf, device):
    results = model.predict(pil, conf=conf, device=device, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return pil, False
    boxes = results[0].boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    cropped = pil.crop((int(x1), int(y1), int(x2), int(y2)))
    if cropped.height > cropped.width:
        cropped = cropped.rotate(-90, expand=True)
    return cropped, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--conf', type=float, default=0.8)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--device', default='cpu', help="'cpu' (default) or a GPU index like '0'")
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    assert os.path.exists(args.weights), f"YOLO weights not found: {args.weights}"
    os.makedirs(args.out_dir, exist_ok=True)
    model = YOLO(args.weights)

    pdfs = sorted(os.path.join(r, f)
                  for r, _, files in os.walk(args.pdf_dir)
                  for f in files if f.lower().endswith('.pdf'))
    assert pdfs, f"No PDFs in {args.pdf_dir}"
    print(f"{len(pdfs)} PDFs -> {args.out_dir} (conf={args.conf}, dpi={args.dpi}, device={args.device})")

    n_crop = n_ok = n_skip = 0
    for p in tqdm(pdfs, desc="crop"):
        stem = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.out_dir, f"{stem}.png")
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue
        try:
            pages = convert_from_path(p, dpi=args.dpi, first_page=1, last_page=1)
            if not pages:
                print(f"no page0: {stem}")
                continue
            cropped, did = crop_one(model, pages[0].convert('RGB'), args.conf, args.device)
            cropped.save(out_path, format='PNG')
            n_ok += 1
            n_crop += int(did)
        except Exception as e:
            print(f"fail {stem}: {e}")

    print(f"\nwrote {n_ok} PNGs ({n_crop} YOLO-cropped, {n_ok - n_crop} no-detection/full-frame), "
          f"{n_skip} skipped (already existed)")
    print(f"Now score with CROP off, pointing the cohort img_dir at: {args.out_dir}")


if __name__ == '__main__':
    main()
