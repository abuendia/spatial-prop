#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0)   # 4 GPUs
JOBS_PER_GPU=1 # how many concurrent jobs per GPU
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/train_gnn_with_celltype.py

DATASETS=("farah")
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"
GENEPT_EMBEDS_PATH="/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/genept_embeds/zenodo/genept_embed/GenePT_gene_embedding_ada_text.pickle"

# (k-hop predict_celltype train_multitask genept_strategy use_oracle_ct ablate_gene_expression use_one_hot_ct, attention_pool, predict_residuals, residual_penalty, debug)
EXPERIMENTS=(
  # "3 True False none False False False center False False False" # 3-hop
  # "2 True False none False False False center True False False" # 2-hop residuals
  # "2 True False none False False False center center False False" # base 2-hop predict with cell type
  # "2 True False none False False False GlobalAttention False False False" # 2-hop global attention
  "3 True False none False False False center False False False" # base 2-hop no cell type (final model)
)

EPOCHS=50
# ----------------

# FIFO as a GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo" 

# seed tokens: each GPU gets $JOBS_PER_GPU slots
for g in "${GPUS[@]}"; do
  for ((i=0; i<JOBS_PER_GPU; i++)); do
    echo "$g" >&3
  done
done

# launch jobs (up to JOBS_PER_GPU per GPU at a time)
for dataset in "${DATASETS[@]}"; do
  for exp_config in "${EXPERIMENTS[@]}"; do
    # Parse tuple
    read -r k_hop predict_celltype train_multitask genept_strategy use_oracle_ct ablate_gene_expression use_one_hot_ct pool predict_residuals residual_penalty debug <<< "$exp_config"
    read -r gpu <&3   # blocks until a GPU is free

    {
      # Build flags and EXP_NAME based on configuration (same logic as run_train_with_celltype.sh)
      if [ "$predict_celltype" = True ]; then
        PREDICT_CELLTYPE_FLAG="--predict_celltype"

        if [ "$train_multitask" = True ]; then
          TRAIN_MULTITASK_FLAG="--train_multitask"
          EXP_NAME="expression_with_celltype_multitask_khop${k_hop}"
        else
          TRAIN_MULTITASK_FLAG=""
          EXP_NAME="expression_with_celltype_decoupled_khop${k_hop}"
        fi
      else
        PREDICT_CELLTYPE_FLAG=""
        TRAIN_MULTITASK_FLAG=""
        EXP_NAME="expression_only_khop${k_hop}"
      fi

      if [ "$genept_strategy" = "early_fusion" ]; then
        GENEPT_STRATEGY_FLAG="--genept_strategy early_fusion"
        GENEPT_EMBEDDINGS_FLAG="--genept_embeddings $GENEPT_EMBEDS_PATH"
        EXP_NAME="${EXP_NAME}_genept_early_fusion"
      elif [ "$genept_strategy" = "late_fusion" ]; then
        GENEPT_STRATEGY_FLAG="--genept_strategy late_fusion"
        GENEPT_EMBEDDINGS_FLAG="--genept_embeddings $GENEPT_EMBEDS_PATH"
        EXP_NAME="${EXP_NAME}_genept_late_fusion"
      elif [ "$genept_strategy" = "xattn" ]; then
        GENEPT_STRATEGY_FLAG="--genept_strategy xattn"
        GENEPT_EMBEDDINGS_FLAG="--genept_embeddings $GENEPT_EMBEDS_PATH"
        EXP_NAME="${EXP_NAME}_genept_xattn"
      elif [ "$genept_strategy" = "none" ]; then
        GENEPT_STRATEGY_FLAG=""
        GENEPT_EMBEDDINGS_FLAG=""
        EXP_NAME="${EXP_NAME}_no_genept"
      fi

      if [ "$debug" = True ]; then
        DEBUG_FLAG="--debug"
        EXP_NAME="${EXP_NAME}_debug"
      else
        DEBUG_FLAG=""
        EXP_NAME="${EXP_NAME}"
      fi

      if [ "$use_oracle_ct" = True ]; then
        USE_ORACLE_CT_FLAG="--use_oracle_ct"
        EXP_NAME="${EXP_NAME}_oracle_ct"
      else
        USE_ORACLE_CT_FLAG=""
        EXP_NAME="${EXP_NAME}"
      fi

      if [ "$ablate_gene_expression" = True ]; then
        ABLATE_GENE_EXPRESSION_FLAG="--ablate_gene_expression"
        EXP_NAME="${EXP_NAME}_ablate_gene_expression"
      else
        ABLATE_GENE_EXPRESSION_FLAG=""
        EXP_NAME="${EXP_NAME}"
      fi

      if [ "$use_one_hot_ct" = True ]; then
        USE_ONE_HOT_CT_FLAG="--use_one_hot_ct"
        EXP_NAME="${EXP_NAME}_one_hot_ct"
      else
        USE_ONE_HOT_CT_FLAG=""
        EXP_NAME="${EXP_NAME}_softmax_ct"
      fi

      if [ "$pool" = "ASAPooling" ]; then
        POOL_FLAG="--pool ASAPooling"
        EXP_NAME="${EXP_NAME}_ASAPooling"
      elif [ "$pool" = "SAGPooling" ]; then
        POOL_FLAG="--pool SAGPooling"
        EXP_NAME="${EXP_NAME}_SAGPooling"
      elif [ "$pool" = "GlobalAttention" ]; then
        POOL_FLAG="--pool GlobalAttention"
        EXP_NAME="${EXP_NAME}_GlobalAttention"
      elif [ "$pool" = "center" ]; then
        POOL_FLAG="--pool center"
        EXP_NAME="${EXP_NAME}_center_pool"
      else
        POOL_FLAG=""
        EXP_NAME="${EXP_NAME}"
      fi

      if [ "$predict_residuals" = True ]; then
        PREDICT_RESIDUALS_FLAG="--predict_residuals"
        EXP_NAME="${EXP_NAME}_predict_residuals"
      else
        PREDICT_RESIDUALS_FLAG=""
        EXP_NAME="${EXP_NAME}"
      fi
      
      if [ "$residual_penalty" = True ]; then
        EXP_NAME="${EXP_NAME}_residual_penalty"
      else
        EXP_NAME="${EXP_NAME}"
      fi

      ts=$(date +%Y%m%d_%H%M%S)
      log="$LOGDIR/residuals_${dataset}_${EXP_NAME}_${ts}.log"
      echo "[$(date +%T)] start $dataset $EXP_NAME on GPU $gpu -> $log"

      # Run and capture both stdout and stderr to the log file
      CUDA_VISIBLE_DEVICES="$gpu" python "$PY" \
        --dataset "$dataset" \
        --base_path "$BASE/data/raw" \
        --exp_name "${EXP_NAME}" \
        --k_hop $k_hop \
        --augment_hop 2 \
        --center_celltypes "all" \
        --node_feature "expression" \
        --inject_feature "none" \
        --learning_rate 0.0001 \
        --loss weightedl1 \
        --epochs $EPOCHS \
        --do_eval \
        $USE_ORACLE_CT_FLAG \
        $GENEPT_STRATEGY_FLAG \
        $GENEPT_EMBEDDINGS_FLAG \
        $TRAIN_MULTITASK_FLAG \
        $PREDICT_CELLTYPE_FLAG \
        $DEBUG_FLAG \
        $ABLATE_GENE_EXPRESSION_FLAG \
        $USE_ONE_HOT_CT_FLAG \
        $POOL_FLAG \
        $PREDICT_RESIDUALS_FLAG \
        >"$log" 2>&1

      status=$?
      echo "$gpu" >&3      # return one GPU slot token
      echo "[$(date +%T)] done  $dataset $EXP_NAME on GPU $gpu (exit $status) | log: $log"
      exit $status
    } &
  done
done

wait
exec 3>&- 3<&-
echo "All datasets finished."
