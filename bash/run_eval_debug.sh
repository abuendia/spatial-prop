#!/bin/bash
set -uo pipefail

# ---- config ----
GPU=0
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/model_performance.py
DATASET=zeng
EXP_NAME=check_eval_model_performance

CUDA_VISIBLE_DEVICES="$GPU" python "$PY" \
        --dataset "$DATASET" \
        --base_path "$BASE/data/raw" \
        --exp_name "${EXP_NAME}" \
        --k_hop 2 \
        --augment_hop 2 \
        --center_celltypes "all" \
        --node_feature "expression" \
        --inject_feature "none" \
        --debug
