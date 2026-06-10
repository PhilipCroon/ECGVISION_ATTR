# train_ECGVISION_ATTR

Retrain the image-based AI-ECG model (ECGVISION / EfficientNetB3 transfer) for
**ATTR cardiac amyloidosis**, refreshed with the March-2026 PyP data update.

This repo mirrors the cohort-building logic of the `multimodal_amyloid` research
repo but is **standalone** — it does not modify that repo. Differences from the
source pipeline are deliberate and listed below.

## What changed vs. the research repo

| Item | Research repo (`multimodal_amyloid`) | This repo |
|------|--------------------------------------|-----------|
| PyP source | `amyloid_pyp_all_07222025.csv` | `amyloid_pyp_all_03102026.parquet` (data update) |
| Train/test cutoff | `2024-07-01` | `2025-07-01` (later → more training data, smaller temporal test) |
| Rationale | internal + external eval | external validation done elsewhere → internal test is a small recent temporal holdout |
| Echo / ECG metadata | re-derived | **reused** from `multimodal_amyloid` (unchanged by a PyP-only update) |

## Cohort logic (unchanged from research repo)

- **Cases (amyloid):** PyP "consistent with TTR cardiac amyloidosis" (+ tafamidis-treated MRNs added as positive).
- **pyp_negative:** PyP "not consistent".
- **control (LVH / AS):** echo-derived `Composite_LVH_binary == 1` or `SevereAS_ObjSubj == 1`, excluding PyP cases.
- **ECG window:** ECG within `[-365d, ∞)` of PyP date (cases) / `±30d` of echo (controls).
- **Matching:** R `MatchIt` 1:10 nearest-neighbor on Age + Sex — amyloid vs. control (single arm). Controls capped at 5 most-recent ECGs/patient; amyloid uncapped.

## Pipeline

```
# 0. (optional, reproducibility only) re-derive echo phenotypes — usually REUSE existing output
python src/clean_echo.py

# 1. build cohort: merge PyP + ECG + echo, date-window, train/test split @ 2025-07-01
python src/build_cohort.py            # -> tabs/cohort_train.csv, tabs/cohort_test.csv

# 2. 1:10 age+sex matching (patient level), expand to ECG level
Rscript src/matching.R                # -> tabs/train_matched_1_10.csv

# 3. train (EfficientNetB3 transfer, contrastive init)
python train/train.py
```

Or: `bash run.sh`

## "More training data" tuning dials (in `src/matching.R`)

Both currently throttle volume — change with counts visible, not silently:
- `slice_head(n = 5)` caps ECGs per patient at 5.
- the `pyp_negative` group is **dropped** by the matching step (only amyloid/control/lvh kept).

## Note on execution

All paths target the Yale compute box (`/home/pmc57/...`, `/mnt/...`). Scripts
are not runnable on a laptop. Verification = sanity-count prints on the server.
