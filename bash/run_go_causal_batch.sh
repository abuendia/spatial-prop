#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0 1 2)   # GPUs to use
JOBS_PER_GPU=1 # how many concurrent jobs per GPU

BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PAIRS_PATH=/oak/stanford/groups/akundaje/abuen/spatial/CausalInteractionBench/pairs
PY=$BASE/src/spatial_gnn/scripts/go_causal_interaction.py
DATASETS=("reprogramming")
MODEL_TYPE=("model" "global_mean" "khop_mean")

LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"
# ----------------

# FIFO as a GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"

# seed tokens: each GPU gets $JOBS_PER_GPU slots
for g in "${GPUS[@]}"; do
  for ((i=0; i<JOBS_PER_GPU; i++)); do
    echo "$g" >&3
  done
done

# launch jobs (up to JOBS_PER_GPU per GPU at a time)
for dataset in "${DATASETS[@]}"; do
  for model_type in "${MODEL_TYPE[@]}"; do
    read -r gpu <&3   # blocks until a GPU slot is free

    {
      ts=$(date +%Y%m%d_%H%M%S)
      log="$LOGDIR/go_itxn_${dataset}_${model_type}_${ts}.log"
      echo "[$(date +%T)] start $dataset ($model_type) on GPU $gpu -> $log"

      # Run and capture both stdout and stderr to the log file
      CUDA_VISIBLE_DEVICES="$gpu" python "$PY" \
        --dataset "$dataset" \
        --base_path "$BASE/data/raw" \
        --k_hop 2 \
        --augment_hop 2 \
        --center_celltypes "all" \
        --node_feature "expression" \
        --inject_feature "none" \
        --perturb_approach "multiplier" \
        --num_props 10 \
        --exp_name "$dataset" \
        --model_type "$model_type" \
        --pairs_path "$PAIRS_PATH" \
        >"$log" 2>&1

      status=$?
      echo "$gpu" >&3      # return one GPU slot token
      echo "[$(date +%T)] done  $dataset ($model_type) on GPU $gpu (exit $status) | log: $log"
      exit $status
    } &
  done
done

wait
exec 3>&- 3<&-
echo "All datasets finished."
