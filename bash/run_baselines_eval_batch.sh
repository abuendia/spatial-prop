#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0 1 2 3)
BASE=.
PY=$BASE/src/spatial_gnn/scripts/run_expression_baselines.py
DATASETS=("aging_coronal" "aging_sagittal" "exercise" "reprogramming" "kukanja" "androvic" "zeng" "pilot" "farah")
BASELINE_TYPE="khop_mean"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"

# FIFO GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo" 
for g in "${GPUS[@]}"; do echo "$g" >&3; done

for dataset in "${DATASETS[@]}"; do
  read -r gpu <&3  # blocks until a GPU is free

  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOGDIR/expression_baselines_${BASELINE_TYPE}_${dataset}_${ts}.log"
    echo "[$(date +%T)] start $dataset on GPU $gpu -> $log"

    CUDA_VISIBLE_DEVICES="$gpu" python "$PY" \
      --dataset "$dataset" \
      --base_path ./data/raw \
      --exp_name "$dataset" \
      --k_hop 2 \
      --augment_hop 2 \
      --center_celltypes "all" \
      --node_feature "expression" \
      --inject_feature "none" \
      --baseline_type "$BASELINE_TYPE" \
      >"$log" 2>&1

    status=$?
    echo "$gpu" >&3 # return GPU token
    echo "[$(date +%T)] done  $dataset on GPU $gpu (exit $status) | log: $log"
    exit $status
  } &
done

wait
echo "All datasets finished."