#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0 1)   # 2 GPUs
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/run_baselines.py
DATASETS=("aging_coronal" "aging_sagittal" "exercise" "reprogramming" "allen" "kukanja" "androvic" "zeng" "pilot" "liverperturb" "lohoff")
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
  read -r gpu <&3   # blocks until a GPU is free

  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOGDIR/baselines_updated_${dataset}_${ts}.log"
    echo "[$(date +%T)] start $dataset on GPU $gpu -> $log"

    # Run and capture both stdout and stderr to the log file
    CUDA_VISIBLE_DEVICES="$gpu" python "$PY" \
      --dataset "$dataset" \
      --base_path /oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw \
      --exp_name "$dataset" \
      --k_hop 2 \
      --augment_hop 2 \
      --center_celltypes "all" \
      --node_feature "expression" \
      --inject_feature "none" \
      >"$log" 2>&1

    status=$?
    echo "$gpu" >&3      # return GPU token
    echo "[$(date +%T)] done  $dataset on GPU $gpu (exit $status) | log: $log"
    exit $status
  } &
done

wait
exec 3>&- 3<&-
echo "All datasets finished."
