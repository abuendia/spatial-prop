#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0 1 3)   # GPUs to use
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PAIRS_PATH=/oak/stanford/groups/akundaje/abuen/spatial/CausalInteractionBench/pairs
PY=$BASE/src/spatial_gnn/scripts/go_causal_interaction.py
DATASETS=("aging_coronal" "aging_sagittal" "exercise" "reprogramming")
# DATASETS=("farah" "kukanja" "androvic" "zeng" "pilot")
MODEL_TYPE=("model" "global_mean" "khop_mean")

LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"
# ----------------

# FIFO as a GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo" 

# seed tokens
for g in "${GPUS[@]}"; do echo "$g" >&3; done

# launch jobs (one per GPU at a time)
for dataset in "${DATASETS[@]}"; do
  for model_type in "${MODEL_TYPE[@]}"; do
    read -r gpu <&3   # blocks until a GPU is free

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
      echo "$gpu" >&3      # return GPU token
      echo "[$(date +%T)] done  $dataset ($model_type) on GPU $gpu (exit $status) | log: $log"
      exit $status
    } &
  done
done

wait
exec 3>&- 3<&-
echo "All datasets finished."
