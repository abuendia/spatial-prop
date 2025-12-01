#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0 1 2 3) 
JOBS_PER_GPU=1 # concurrent jobs per GPU
BASE=.
PAIRS_PATH=../CausalInteractionBench/pairs
PY=$BASE/src/spatial_gnn/scripts/run_go_causal_eval.py
DATASETS=("aging_coronal" "aging_sagittal" "exercise" "reprogramming" "kukanja" "androvic" "zeng" "pilot" "farah")
MODEL_TYPE=("model" "khop_mean" "global_mean")
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"

# FIFO GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"
for g in "${GPUS[@]}"; do
  for ((i=0; i<JOBS_PER_GPU; i++)); do
    echo "$g" >&3
  done
done

for dataset in "${DATASETS[@]}"; do
  for model_type in "${MODEL_TYPE[@]}"; do
    read -r gpu <&3   # blocks until a GPU slot is free

    {
      ts=$(date +%Y%m%d_%H%M%S)
      log="$LOGDIR/go_causal_eval_${dataset}_${model_type}_${ts}.log"
      echo "[$(date +%T)] start $dataset ($model_type) on GPU $gpu -> $log"

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
        --model_path "$BASE/results/expr_model_predict/appendix/expression_only_khop2_no_genept_softmax_ct_center_pool/${dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04/model.pth" \
        >"$log" 2>&1

      status=$?
      echo "$gpu" >&3
      echo "[$(date +%T)] done  $dataset ($model_type) on GPU $gpu (exit $status) | log: $log"
      exit $status
    } &
  done
done

wait
echo "All datasets finished."
