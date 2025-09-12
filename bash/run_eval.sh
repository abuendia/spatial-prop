dataset="aging_sagittal"

CUDA_VISIBLE_DEVICES=0 python /oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/scripts/model_performance.py \
    --dataset "$dataset" \
    --base_path /oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw \
    --exp_name "$dataset" \
    --k_hop 2 \
    --augment_hop 2 \
    --center_celltypes "all" \
    --node_feature "expression" \
    --inject_feature "none" \
    --learning_rate 0.0001 \
    --loss "weightedl1" \
    --epochs 50 
