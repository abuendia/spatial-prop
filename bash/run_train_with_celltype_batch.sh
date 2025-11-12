#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0 1 2 3)   # 4 GPUs
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/train_gnn_with_celltype.py
DATASETS_FIRST_HALF=("aging_coronal" "aging_sagittal"  "zeng" "pilot")
DATASETS_SECOND_HALF=("exercise" "reprogramming" "kukanja" "androvic")

DATASETS=DATASETS_SECOND_HALF # (krishna, shakti)
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"
GENEPT_EMBEDS_PATH="/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/genept_embeds/zenodo/genept_embed/GenePT_gene_embedding_ada_text.pickle"

# Configuration arrays for different flag combinations
TRAIN_MULTITASK_VALS=(False)
GENEPT_STRATEGY_VALS=(none)
DEBUG_VALS=(False)
PREDICT_CELLTYPE_VALS=(True False)
USE_ORACLE_CT_VALS=(False True)
ABLATE_GENE_EXPRESSION_VALS=(False True)
USE_ONE_HOT_CT_VALS=(False True)
EPOCHS=50
# ----------------

# FIFO as a GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo" 

# seed tokens
for g in "${GPUS[@]}"; do echo "$g" >&3; done

# launch jobs (one per GPU at a time)
for dataset in "${DATASETS[@]}"; do
  for predict_celltype in "${PREDICT_CELLTYPE_VALS[@]}"; do
    for train_multitask in "${TRAIN_MULTITASK_VALS[@]}"; do
      for genept_strategy in "${GENEPT_STRATEGY_VALS[@]}"; do
        for debug in "${DEBUG_VALS[@]}"; do
          for use_oracle_ct in "${USE_ORACLE_CT_VALS[@]}"; do
            for ablate_gene_expression in "${ABLATE_GENE_EXPRESSION_VALS[@]}"; do
              for use_one_hot_ct in "${USE_ONE_HOT_CT_VALS[@]}"; do
                # Skip if both ablate_gene_expression and use_oracle_ct are False
                if [ "$ablate_gene_expression" = True ] && [ "$use_oracle_ct" = False ]; then
                  continue
                fi
                read -r gpu <&3   # blocks until a GPU is free

                {
                  # Build flags and EXP_NAME based on configuration (same logic as run_train_with_celltype.sh)
                  if [ "$predict_celltype" = True ]; then
                    PREDICT_CELLTYPE_FLAG="--predict_celltype"

                    if [ "$train_multitask" = True ]; then
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
                  ts=$(date +%Y%m%d_%H%M%S)
                  log="$LOGDIR/baselines_${dataset}_${EXP_NAME}_${ts}.log"
                  echo "[$(date +%T)] start $dataset $EXP_NAME on GPU $gpu -> $log"

                  # Run and capture both stdout and stderr to the log file
                  CUDA_VISIBLE_DEVICES="$gpu" python "$PY" \
                    --dataset "$dataset" \
                    --base_path "$BASE/data/raw" \
                    --exp_name "${EXP_NAME}" \
                    --k_hop 2 \
                    --augment_hop 2 \
                    --center_celltypes "all" \
                    --node_feature "expression" \
                    --inject_feature "none" \
                    --learning_rate 0.0001 \
                    --loss weightedl1 \
                    --epochs $EPOCHS \
                    $USE_ORACLE_CT_FLAG \
                    $GENEPT_STRATEGY_FLAG \
                    $GENEPT_EMBEDDINGS_FLAG \
                    $TRAIN_MULTITASK_FLAG \
                    $PREDICT_CELLTYPE_FLAG \
                    $DEBUG_FLAG \
                    $ABLATE_GENE_EXPRESSION_FLAG \
                    $USE_ONE_HOT_CT_FLAG \
                    >"$log" 2>&1

                  status=$?
                  echo "$gpu" >&3      # return GPU token
                  echo "[$(date +%T)] done  $dataset $EXP_NAME on GPU $gpu (exit $status) | log: $log"
                  exit $status
                } &
              done
            done
          done
        done
      done
    done
  done
done

wait
exec 3>&- 3<&-
echo "All datasets finished."
