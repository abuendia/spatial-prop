#!/bin/bash
#SBATCH -J k2_wL14_E # job name
#SBATCH -G 1
#SBATCH --cpus-per-gpu=8
#SBATCH -t 10080 # Runtime in minutes 
#SBATCH -p jamesz # Partition to submit to 
#SBATCH --account=jamesz
#SBATCH --mem=16G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH --exclude=sh03-18n12
#SBATCH -o k2_wL14_E_%A.out # Standard out goes to this file 
#SBATCH -e k2_wL14_E_%A.err # Standard err goes to this filehostname 

source activate merfish_gnn
python scripts/train_gnn_model_expression.py --dataset aging_coronal --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 10
conda deactivate