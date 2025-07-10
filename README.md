# spatial-gnn: Predicting the spatial effects of single-cell genetic perturbations

## Training GNN

    python scripts/train_gnn_model_expression.py \
        --dataset aging_coronal \
        --base_path /oak/stanford/groups/jamesz/abuen/spatial-sc/data/raw \
        --k_hop 2 \
        --augment_hop 2 \
        --center_celltypes "T cell,NSC,Pericyte" \
        --node_feature "expression" \
        --inject_feature "none" \
        --learning_rate 0.0001 \
        --loss "weightedl1" \
        --epochs 50
