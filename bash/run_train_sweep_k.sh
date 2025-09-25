#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0 1 2)   # 2 GPUs
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/train_gnn_model_expression.py
DATASETS=("aging_coronal" "aging_sagittal" "exercise" "reprogramming" "allen" "kukanja" "androvic" "zeng" "pilot" "liverperturb" "lohoff")
K_HOP=(3 4 5)
# ----------------

# FIFO as a GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo" 

# seed tokens
for g in "${GPUS[@]}"; do echo "$g" >&3; done

# launch jobs (one per GPU at a time)
for k_hop in "${K_HOP[@]}"; do
  for dataset in "${DATASETS[@]}"; do

    read -r gpu <&3   # blocks until a GPU is free

    {
      echo "[$(date +%T)] start $dataset on GPU $gpu | k_hop: $k_hop"

      # Run training script (logging is handled internally by the Python script)
      CUDA_VISIBLE_DEVICES="$gpu" python "$PY" \
        --dataset "$dataset" \
        --base_path "$BASE/data/raw" \
        --k_hop $k_hop \
        --augment_hop 2 \
        --center_celltypes all \
        --node_feature expression \
        --inject_feature none \
        --learning_rate 0.0001 \
        --loss weightedl1 \
        --epochs 100 \
        --exp_name "benchmark_base_${dataset}_k${k_hop}" \
        --do_eval

      status=$?
      echo "$gpu" >&3      # return GPU token
      echo "[$(date +%T)] done  $dataset on GPU $gpu (exit $status) | k_hop: $k_hop"
      exit $status
    } &
  done
done

wait
exec 3>&- 3<&-
echo "All datasets finished."
