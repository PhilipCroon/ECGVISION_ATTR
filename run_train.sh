#!/bin/bash
# Train an ENSEMBLE: one model per SEED, each on a different bagging split + RNG
# (see train/train.py:71-83). Soft-average the checkpoints' probabilities at eval
# time to (hopefully) beat the single-model AUROC.
#
# Two modes:
#   SEQUENTIAL (default) — all seeds run one-after-another on a single GPU.
#   PARALLEL             — set GPUS to fan seeds out across GPUs, one proc per GPU.
#
# Usage:
#   bash run_train.sh                              # seeds 1 2 3, sequential, GPU 1
#   CUDA_VISIBLE_DEVICES=2 bash run_train.sh        # sequential on GPU 2
#   GPUS="1 2 3" bash run_train.sh                  # PARALLEL: s1->gpu1, s2->gpu2, s3->gpu3
#   GPUS="4 5" SEEDS="1 2" bash run_train.sh        # 2 seeds across 2 GPUs
#   SEEDS="4 5 6 7" bash run_train.sh               # custom seeds (sequential)
#   SEEDS="" bash run_train.sh                      # single non-seeded run (legacy)
#   RUN_TAG="ens_b256" bash run_train.sh            # free-text tag -> run_registry.csv
#
# Overnight, detached (survives SSH drop):
#   GPUS="1 2 3" nohup bash run_train.sh > /dev/null 2>&1 &
#   tail -f logs/ensemble3_*_master.log
#
# !! PARALLEL prerequisites — read before using GPUS:
#   1. HOST RAM: each proc RAM-caches the FULL train image set independently. N procs
#      = N copies in RAM (worse if NFORMATS>1). Confirm free RAM >= N x one run, or
#      you get a host-RAM OOM (process 'Killed'), not a GPU OOM.
#   2. PNG CACHE must be WARM. The first-ever render writes PNGs to a shared dir; N
#      procs rendering the same files at once corrupts the cache. If your cache is
#      already warm from a prior run, just launch parallel (default). If it is COLD,
#      set PREWARM=1: this runs the FIRST seed completely on its own (train.py has no
#      render-only mode, so the render rides along with a full training run), which
#      populates the cache; seeds 2..N then fan out in parallel. Net cold-cache
#      walltime is therefore ~2x one run, not 1x. Warm cache + no PREWARM = ~1x.
#      NOTE: PREWARM only fully covers the cache when NFORMATS=1. With NFORMATS>1 the
#      per-seed bagging splits differ, so the val->train swing set lacks pre-rendered
#      variants and seeds 2..N can still race. Use NFORMATS=1 for parallel, or
#      pre-render NFORMATS variants once on a single proc before launching.
#
# Each seed -> own per-seed log in logs/. Best val_auroc + ckpt path are recorded by
# train.py into models/run_registry.csv (one row per run_id, suffix _s1/_s2/_s3).
set -uo pipefail   # NOT -e: one seed failing must not abort the rest of the sweep
cd "$(dirname "${BASH_SOURCE[0]}")"

SEEDS="${SEEDS-1 2 3}"
GPUS="${GPUS:-}"                       # non-empty => PARALLEL mode
RUN_TAG="${RUN_TAG:-ensemble3}"
PREWARM="${PREWARM:-0}"
export RUN_TAG

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/${RUN_TAG}_${STAMP}_master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1     # mirror everything below into the master log

if [ -n "$GPUS" ]; then MODE="PARALLEL (gpus: $GPUS)"; else MODE="SEQUENTIAL (gpu: ${CUDA_VISIBLE_DEVICES:-1})"; fi
echo "=================================================================="
echo " ENSEMBLE TRAIN  tag=$RUN_TAG"
echo " mode:  $MODE"
echo " seeds: ${SEEDS:-<none / single run>}"
echo " host:  $(hostname)   start: $(date)"
echo " master log: $MASTER_LOG"
echo "=================================================================="

gpu_snapshot() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu \
             --format=csv 2>/dev/null || echo "  (nvidia-smi query failed)"
}

# run_one <seed> <gpu>  — gpu='' keeps the inherited CUDA_VISIBLE_DEVICES.
run_one() {
  local seed="$1" gpu="$2" label per_log start_ts end_ts rc
  if [ -n "$seed" ]; then label="s${seed}"; else label="noseed"; fi
  [ -n "$gpu" ] && label="${label}_gpu${gpu}"
  per_log="$LOG_DIR/${RUN_TAG}_${STAMP}_${label}.log"

  echo ">>> [$label] START $(date)  -> $per_log"
  start_ts=$(date +%s)
  (
    [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
    [ -n "$seed" ] && export SEED="$seed"
    python train/train.py
  ) > "$per_log" 2>&1
  rc=$?
  end_ts=$(date +%s)
  echo ">>> [$label] END   $(date)  exit=$rc  elapsed=$(( (end_ts - start_ts) / 60 ))min"
  if [ "$rc" -ne 0 ]; then
    echo ">>> [$label] FAILED (exit $rc). Check $per_log:"
    echo "    'ResourceExhaustedError' = GPU VRAM OOM ; 'Killed'/signal = host RAM OOM."
  fi
  return "$rc"
}

# Optional serial pre-warm so parallel procs never race on a cold PNG cache.
prewarm() {
  echo ">>> PREWARM: running first seed solo (renders cache + trains it; ~1 full run)..."
  local first_seed; first_seed=$(echo $SEEDS | awk '{print $1}')
  local gpu0; gpu0=$(echo $GPUS | awk '{print $1}')
  ( [ -n "$gpu0" ] && export CUDA_VISIBLE_DEVICES="$gpu0"
    [ -n "$first_seed" ] && export SEED="$first_seed"
    python train/train.py ) > "$LOG_DIR/${RUN_TAG}_${STAMP}_prewarm.log" 2>&1
  echo ">>> PREWARM done (this also trained seed=$first_seed; it is NOT re-run below)."
}

gpu_snapshot
declare -a RESULTS
overall_rc=0

read -r -a SEED_ARR <<< "$SEEDS"
read -r -a GPU_ARR  <<< "$GPUS"

if [ -z "${SEEDS// /}" ]; then
  # ---- single legacy non-seeded run ----
  run_one "" "${CUDA_VISIBLE_DEVICES:-}"; rc=$?
  RESULTS+=("noseed:exit=$rc"); [ "$rc" -ne 0 ] && overall_rc=1

elif [ -z "$GPUS" ]; then
  # ---- SEQUENTIAL: all seeds on the inherited GPU ----
  for s in "${SEED_ARR[@]}"; do
    run_one "$s" ""; rc=$?
    RESULTS+=("s${s}:exit=$rc"); [ "$rc" -ne 0 ] && overall_rc=1
  done

else
  # ---- PARALLEL: zip seeds <-> gpus, one background proc per pair ----
  if [ "${#GPU_ARR[@]}" -lt "${#SEED_ARR[@]}" ]; then
    echo "WARNING: ${#SEED_ARR[@]} seeds but only ${#GPU_ARR[@]} GPUs."
    echo "         Seeds beyond the GPU count would share a GPU (OOM risk) -> SKIPPED."
    echo "         Give one GPU per seed, or run sequentially."
  fi
  start_idx=0
  if [ "$PREWARM" = "1" ]; then
    prewarm; pw_rc=$?
    RESULTS+=("s${SEED_ARR[0]}(prewarm):exit=$pw_rc")
    [ "$pw_rc" -ne 0 ] && overall_rc=1
    start_idx=1   # seed 0 already trained during pre-warm
  fi
  declare -a PIDS LABELS
  n=${#SEED_ARR[@]}
  for ((i=start_idx; i<n && i<${#GPU_ARR[@]}; i++)); do
    s="${SEED_ARR[$i]}"; g="${GPU_ARR[$i]}"
    run_one "$s" "$g" &
    PIDS+=("$!"); LABELS+=("s${s}_gpu${g}")
    echo ">>> launched s${s} on gpu${g} (pid $!)"
  done
  for ((j=0; j<${#PIDS[@]}; j++)); do
    wait "${PIDS[$j]}"; rc=$?
    RESULTS+=("${LABELS[$j]}:exit=$rc"); [ "$rc" -ne 0 ] && overall_rc=1
  done
fi

echo ""
echo "=================================================================="
echo " ENSEMBLE DONE  $(date)"
for r in "${RESULTS[@]}"; do echo "   $r"; done
echo "   registry: models/run_registry.csv  (best val_auroc per run_id)"
echo "   per-seed logs: $LOG_DIR/${RUN_TAG}_${STAMP}_*.log"
echo "   overall exit: $overall_rc"
echo "=================================================================="
exit "$overall_rc"
