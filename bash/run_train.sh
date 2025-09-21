dataset="aging_coronal"
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
# Train with config
CUDA_VISIBLE_DEVICES=0 python $BASE/src/spatial_gnn/scripts/train_gnn_model_expression.py \
      --dataset "$dataset" \
      --base_path "$BASE/data/raw" \
      --k_hop 2 \
      --augment_hop 2 \
      --center_celltypes all \
      --node_feature expression \
      --inject_feature none \
      --learning_rate 0.0001 \
      --loss weightedl1 \
      --epochs 1 \
      --exp_name "benchmark_base_${dataset}" \
      --do_eval \
      --debug \
      --genept_embeddings "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/genept_embeds/zenodo/genept_embed/GenePT_gene_embedding_ada_text.pickle" 
