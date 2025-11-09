genept_strategy="early_fusion"
dataset="reprogramming"
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
# Train with config
TRAIN_MULTITASK=False
if [ "$TRAIN_MULTITASK" = True ]; then
  TRAIN_MULTITASK_FLAG="--train_multitask"
else
  TRAIN_MULTITASK_FLAG=""
fi

exp_name="celltype_multitask_${TRAIN_MULTITASK_FLAG}"

CUDA_VISIBLE_DEVICES=0 python $BASE/src/spatial_gnn/scripts/train_gnn_with_celltype.py \
      --dataset "$dataset" \
      --base_path "$BASE/data/raw" \
      --k_hop 2 \
      --augment_hop 2 \
      --center_celltypes all \
      --node_feature expression \
      --inject_feature none \
      --learning_rate 0.0001 \
      --loss weightedl1 \
      --epochs 50 \
      --exp_name "${exp_name}" \
      --do_eval \
      $TRAIN_MULTITASK_FLAG \
      --predict_celltype
      # --genept_embeddings "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/genept_embeds/zenodo/genept_embed/GenePT_gene_embedding_ada_text.pickle" \
      # --genept_strategy "$genept_strategy" \

