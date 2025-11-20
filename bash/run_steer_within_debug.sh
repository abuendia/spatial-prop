#!/bin/bash
set -uo pipefail

# ---- config ----
GPU=0
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/steering_perturbation_within_baselines.py
DATASET=reprogramming
EXP_NAME=debug_steer_within
EPOCHS=2


CUDA_VISIBLE_DEVICES="$GPU" python "$PY" \
        --dataset "$DATASET" \
        --base_path "$BASE/data/raw" \
        --k_hop 2 \
        --augment_hop 2 \
        --center_celltypes "all" \
        --node_feature "expression" \
        --inject_feature "none" \
        --learning_rate 0.0001 \
        --loss weightedl1 \
        --epochs $EPOCHS \
        --steering_approach "batch_steer_cell" \
        --num_props 10 \
        --exp_name "$EXP_NAME" \
        --debug
