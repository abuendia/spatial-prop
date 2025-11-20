BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/run_baselines.py
BASELINE_TYPE="khop_mean"

CUDA_VISIBLE_DEVICES=0 python "$PY" \
    --dataset "zeng" \
    --base_path /oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw \
    --exp_name "zeng" \
    --k_hop 2 \
    --augment_hop 2 \
    --center_celltypes "all" \
    --node_feature "expression" \
    --inject_feature "none" \
    --debug \
    --baseline_type "$BASELINE_TYPE"

