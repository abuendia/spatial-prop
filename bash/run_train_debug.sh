#!/bin/bash
set -uo pipefail

# ---- config ----
GPU=0
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/train_gnn_with_celltype.py
DATASET=reprogramming
EPOCHS=2
POOL="GlobalAttention"
EXP_NAME=debug_attention_pool_${POOL}

CUDA_VISIBLE_DEVICES="$GPU" python "$PY" \
        --dataset "$DATASET" \
        --base_path "$BASE/data/raw" \
        --exp_name "${EXP_NAME}" \
        --k_hop 2 \
        --augment_hop 2 \
        --center_celltypes "all" \
        --pool "$POOL" \
        --node_feature "expression" \
        --inject_feature "none" \
        --learning_rate 0.0001 \
        --loss weightedl1 \
        --epochs $EPOCHS \
        --do_eval \
        --log_to_terminal \
        --debug