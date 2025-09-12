dataset="aging_coronal"

CUDA_VISIBLE_DEVICES=1 python /oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/scripts/run_baselines.py \
    --dataset "$dataset" \
    --base_path /oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw \
    --exp_name "$dataset" \
    --k_hop 2 \
    --augment_hop 2 \
    --center_celltypes "all" \
    --node_feature "expression" \
    --inject_feature "none" 
