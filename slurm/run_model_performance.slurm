#!/bin/bash
#SBATCH -J model_perf # job name
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 600 # Runtime in minutes 
#SBATCH -p jamesz # Partition to submit to 
#SBATCH --account=jamesz
#SBATCH --nodelist=sh03-10n67,sh03-10n68
#SBATCH --mem=100G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o model_perf_%A.out # Standard out goes to this file 
#SBATCH -e model_perf_%A.err # Standard err goes to this filehostname 

source activate merfish_gnn

python scripts/model_performance.py --dataset aging_coronal --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50

python scripts/model_performance.py --dataset liverperturb --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NK cell,B cell" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50

python scripts/model_performance.py --dataset lohoff --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "all" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50

conda deactivate