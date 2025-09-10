# Train with config
CUDA_VISIBLE_DEVICES=0 python /oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/src/spatial_gnn/scripts/train_gnn_model_expression.py \
    --dataset aging_coronal \
    --base_path /oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/raw \
    --k_hop 2 \
    --augment_hop 2 \
    --center_celltypes "T cell,NSC,Pericyte" \
    --node_feature "expression" \
    --inject_feature "none" \
    --learning_rate 0.0001 \
    --loss "weightedl1" \
    --epochs 50 \
    --exp_name "benchmark_base" \
    --do_eval
