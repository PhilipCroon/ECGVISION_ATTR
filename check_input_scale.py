"""
Verify pixel-value range through contrastive-training path vs current inference path.
Run on the server: python check_input_scale.py
"""
import sys
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.transform import resize

sys.path.append(os.path.dirname(__file__))
import project_constants as project
from utils import load_image_from_disk, make_plot

# --- synthetic test (no real files needed) ---
print("=== Synthetic test ===")
arr = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)

# contrastive training path: DataSequenceTrain_RAM
rot = ndimage.rotate(arr, 5.0, reshape=True, mode='nearest')
out_train = resize(rot, (300, 300))
print(f"TRAIN (ndimage.rotate + skimage.resize): dtype={out_train.dtype}  "
      f"min={out_train.min():.4f}  max={out_train.max():.4f}  mean={out_train.mean():.4f}")

# current inference path: load_image_from_disk
img_pil = Image.fromarray(arr).convert('RGB').resize((300, 300))
out_infer = np.array(img_pil).astype(np.float32)
print(f"INFER (load_image_from_disk, current): dtype={out_infer.dtype}  "
      f"min={out_infer.min():.4f}  max={out_infer.max():.4f}  mean={out_infer.mean():.4f}")

out_infer_div = out_infer / 255.0
print(f"INFER (/255 proposed fix):              dtype={out_infer_div.dtype}  "
      f"min={out_infer_div.min():.4f}  max={out_infer_div.max():.4f}  mean={out_infer_div.mean():.4f}")

# --- real file test (uses first available image from IMAGE_DIR) ---
print("\n=== Real file test ===")
image_dir = project.image_dir
try:
    files = [f for f in os.listdir(image_dir) if f.endswith('.png')][:1]
    if not files:
        print("No PNG files found in IMAGE_DIR — skipping real file test")
    else:
        fid = files[0].replace('.png', '')
        img = load_image_from_disk(fid, image_dir)
        if img is not None:
            print(f"load_image_from_disk: dtype={img.dtype}  "
                  f"min={img.min():.4f}  max={img.max():.4f}  mean={img.mean():.4f}")
        else:
            print("load_image_from_disk returned None")
except Exception as e:
    print(f"Real file test failed: {e}")

# --- verdict ---
print("\n=== Verdict ===")
train_range = out_train.max() - out_train.min()
infer_range = out_infer.max() - out_infer.min()
if infer_range > 10 * train_range:
    print("MISMATCH: inference ~[0,255], training ~[0,1] → fix: divide load_image_from_disk by 255")
elif abs(out_train.mean() - out_infer.mean()) < 0.1:
    print("MATCH: ranges look similar — scale is NOT the problem")
else:
    print(f"UNCERTAIN: train mean={out_train.mean():.3f}, infer mean={out_infer.mean():.3f} — inspect manually")
