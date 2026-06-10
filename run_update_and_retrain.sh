#!/bin/bash
# Full pipeline to retrain on the refreshed pdcfs1 data:
#   1. regenerate the flagged ECG-metadata key (pulls new pdcfs1 through ~present)
#   2. rebuild cohort (new ECGs flow into train/test)
#   3. 1:10 age+sex matching
#   4. retrain (renders all PNGs incl. new pdcfs1 on first pass, then trains)
#
# Run AFTER run_test_comparison.sh and after check_pdcfs1.py is green.
#   bash run_update_and_retrain.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"
LOG="retrain_$(date +%Y%m%d_%H%M%S).log"

step () {
  echo -e "\n========================================================================" | tee -a "$LOG"
  echo "=== $* ===" | tee -a "$LOG"
  echo "========================================================================" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

echo "Pipeline start $(date)  CUDA=$CUDA_VISIBLE_DEVICES  log=$LOG" | tee "$LOG"

step python src/build_ecg_metadata.py     # -> data_LD/ecg_metadata_flagged_..._<today>.csv (auto-picked)
step python src/build_cohort.py            # watch amyloid TRAIN/TEST MRN counts grow vs before
step Rscript src/matching.R                # -> tabs/train_matched_1_10.csv
step python train/train.py                 # watch the [train] n_new tripwire

echo -e "\nDone $(date). New checkpoint: models/attr_amyloid_<today>_unfrozen_<NN>" | tee -a "$LOG"
echo "Next: point NEW_MODEL in train/test_both_models.py at it, then bash run_test_comparison.sh" | tee -a "$LOG"
