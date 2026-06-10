#!/bin/bash
# Train only (metadata/cohort/matching already done). Single GPU 7 -> default
# strategy, no NCCL. Override the GPU with: CUDA_VISIBLE_DEVICES=N bash run_train.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
LOG="train_$(date +%Y%m%d_%H%M%S).log"

echo "Training on GPU(s) $CUDA_VISIBLE_DEVICES  log=$LOG"
python train/train.py 2>&1 | tee "$LOG"
echo "Done. New checkpoint: models/attr_amyloid_<today>_unfrozen_<NN>  (log: $LOG)"
