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
from utils import load_image_from_disk, make_plot, load_images_parallel, DataSequenceAugRAM

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
out_infer = np.array(img_pil).astype(np.float32) / 255.0
print(f"INFER (load_image_from_disk /255):       dtype={out_infer.dtype}  "
      f"min={out_infer.min():.4f}  max={out_infer.max():.4f}  mean={out_infer.mean():.4f}")

# DataSequenceAugRAM path: load_image_from_disk + ndimage.rotate
out_aug = ndimage.rotate(out_infer, 5.0, reshape=False, mode='nearest').astype(np.float32)
print(f"DataSequenceAugRAM (after rotation):     dtype={out_aug.dtype}  "
      f"min={out_aug.min():.4f}  max={out_aug.max():.4f}  mean={out_aug.mean():.4f}")

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
            print(f"load_image_from_disk:                    dtype={img.dtype}  "
                  f"min={img.min():.4f}  max={img.max():.4f}  mean={img.mean():.4f}")
        else:
            print("load_image_from_disk returned None")
except Exception as e:
    print(f"Real file test failed: {e}")

# --- DataSequenceAugRAM end-to-end test ---
print("\n=== DataSequenceAugRAM batch test ===")
try:
    import pandas as pd
    cohort = pd.read_csv(os.path.join(project.tabs_path, 'train_matched_1_10.csv'))
    cohort['amyloid'] = (cohort['group'] == 'amyloid').astype(float)
    cohort['fileID'] = cohort['FileID'].astype(str).str.replace('.dcm$', '', regex=True)
    sample = cohort.head(64).copy()
    imgs = load_images_parallel(sample['fileID'], image_dir)
    sample['image_array'] = sample['fileID'].map(imgs)
    sample = sample[sample['image_array'].notna()].reset_index(drop=True)
    if len(sample) == 0:
        print("No images loaded — skipping DataSequenceAugRAM test")
    else:
        seq = DataSequenceAugRAM(df=sample, batch_size=32, label='amyloid')
        batch_x, batch_y = seq[0]
        print(f"DataSequenceAugRAM batch output:         dtype={batch_x.dtype}  "
              f"min={batch_x.min():.4f}  max={batch_x.max():.4f}  mean={batch_x.mean():.4f}")
except Exception as e:
    print(f"DataSequenceAugRAM test failed: {e}")

# --- verdict ---
print("\n=== Verdict ===")
if out_aug.max() > 10:
    print("MISMATCH: DataSequenceAugRAM output ~[0,255] — scale bug in augmentation path")
elif abs(out_train.mean() - out_aug.mean()) > 0.1:
    print(f"UNCERTAIN: train mean={out_train.mean():.3f}, aug mean={out_aug.mean():.3f} — inspect manually")
else:
    print("MATCH: DataSequenceAugRAM output ~[0,1] — scale looks correct")
