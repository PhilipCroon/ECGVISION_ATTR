#!/bin/bash
# Compare the OLD model vs the 2026-06-05 model on the test set, BEFORE retraining.
# Renders test ECGs with the fixed standard layout (deterministic), scores both on
# identical clean images, prints a patient-level AUROC/AUPRC table.
#
#   OLD = multimodal_amyloid/.../trained_model_Amyloidosis_stage2_age_sex_1_10_15
#   NEW = attr_amyloid_2026_06_05_unfrozen_06   (the model you trained)
#
# Both baked into train/test_both_models.py. Run:  bash run_test_comparison.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"
LOG="test_comparison_$(date +%Y%m%d_%H%M%S).log"

echo "=== OLD vs attr_amyloid_2026_06_05_unfrozen_06 (standard-layout test render) ===" | tee "$LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
python train/test_both_models.py 2>&1 | tee -a "$LOG"
echo -e "\nSaved log: $LOG"
