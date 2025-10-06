#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(1)
BASE=/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn
PY=$BASE/src/spatial_gnn/scripts/train_gnn_model_expression.py
DATASETS=("aging_sagittal" "allen" "androvic" "exercise" "kukanja" "lohoff" "liverperturb" "pilot" "reprogramming" "zeng" "aging_coronal")
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
  read -r gpu <&3   # blocks until a GPU is free

  {
    echo "[$(date +%T)] start $dataset on GPU $gpu"

    CUDA_VISIBLE_DEVICES="$gpu" python "$PY" \
      --dataset "$dataset" \
      --base_path "$BASE/data/raw" \
      --k_hop 3 \
      --augment_hop 2 \
      --center_celltypes all \
      --node_feature expression \
      --inject_feature none \
      --learning_rate 0.0001 \
      --loss weightedl1 \
      --epochs 50 \
      --exp_name xattn \
      --do_eval \
      --genept_embeddings "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/genept_embeds/zenodo/genept_embed/GenePT_gene_embedding_ada_text.pickle" 
    
    status=$?
    echo "$gpu" >&3      # return GPU token
    echo "[$(date +%T)] done  $dataset on GPU $gpu (exit $status)"
    exit $status
  } &
done

wait
exec 3>&- 3<&-
echo "All datasets finished."
