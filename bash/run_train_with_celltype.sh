# Usage: bash run_train_with_celltype.sh <dataset> <predict_celltype> <train_multitask> <genept_strategy> <epochs> <debug>
DATASET=$1
PREDICT_CELLTYPE=$2
TRAIN_MULTITASK=$3
GENEPT_STRATEGY=$4
EPOCHS=$5
DEBUG=$6
GPU=$7
GENEPT_EMBEDS_PATH="/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/genept_embeds/zenodo/genept_embed/GenePT_gene_embedding_ada_text.pickle"
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn


if [ "$PREDICT_CELLTYPE" = True ]; then
  PREDICT_CELLTYPE_FLAG="--predict_celltype"

  if [ "$TRAIN_MULTITASK" = True ]; then
    TRAIN_MULTITASK_FLAG="--train_multitask"
    EXP_NAME="expression_with_celltype_multitask"
  else
    TRAIN_MULTITASK_FLAG=""
    EXP_NAME="expression_with_celltype_decoupled"
  fi
else
  PREDICT_CELLTYPE_FLAG=""
  TRAIN_MULTITASK_FLAG=""
  EXP_NAME="expression_only"
fi

if [ "$GENEPT_STRATEGY" = "early_fusion" ]; then
  GENEPT_STRATEGY_FLAG="--genept_strategy early_fusion"
  GENEPT_EMBEDDINGS_FLAG="--genept_embeddings $GENEPT_EMBEDS_PATH"
  EXP_NAME="${EXP_NAME}_genept_early_fusion"
elif [ "$GENEPT_STRATEGY" = "late_fusion" ]; then
  GENEPT_STRATEGY_FLAG="--genept_strategy late_fusion"
  GENEPT_EMBEDDINGS_FLAG="--genept_embeddings $GENEPT_EMBEDS_PATH"
  EXP_NAME="${EXP_NAME}_genept_late_fusion"
elif [ "$GENEPT_STRATEGY" = "xattn" ]; then
  GENEPT_STRATEGY_FLAG="--genept_strategy xattn"
  GENEPT_EMBEDDINGS_FLAG="--genept_embeddings $GENEPT_EMBEDS_PATH"
  EXP_NAME="${EXP_NAME}_genept_xattn"
elif [ "$GENEPT_STRATEGY" = "none" ]; then
  GENEPT_STRATEGY_FLAG=""
  GENEPT_EMBEDDINGS_FLAG=""
  EXP_NAME="${EXP_NAME}_no_genept"
fi

if [ "$DEBUG" = True ]; then
  DEBUG_FLAG="--debug"
  EXP_NAME="${EXP_NAME}_debug"
else
  DEBUG_FLAG=""
  EXP_NAME="${EXP_NAME}"
fi

echo "EXP_NAME: $EXP_NAME"


CUDA_VISIBLE_DEVICES="$GPU" python $BASE/src/spatial_gnn/scripts/train_gnn_with_celltype.py \
      --dataset "$DATASET" \
      --base_path "$BASE/data/raw" \
      --k_hop 2 \
      --augment_hop 2 \
      --center_celltypes all \
      --node_feature expression \
      --inject_feature none \
      --learning_rate 0.0001 \
      --loss weightedl1 \
      --epochs $EPOCHS \
      --exp_name "${EXP_NAME}" \
      --do_eval \
      $GENEPT_STRATEGY_FLAG \
      $GENEPT_EMBEDDINGS_FLAG \
      $TRAIN_MULTITASK_FLAG \
      $PREDICT_CELLTYPE_FLAG \
      $DEBUG_FLAG
