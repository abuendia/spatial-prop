#!/bin/bash
#SBATCH -J steer_within # job name
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 10080 # Runtime in minutes 
#SBATCH -p jamesz # Partition to submit to 
#SBATCH --account=jamesz
#SBATCH --nodelist=sh03-10n67,sh03-10n68
#SBATCH --mem=100G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o steer_within_%A.out # Standard out goes to this file 
#SBATCH -e steer_within_%A.err # Standard err goes to this filehostname 

source activate merfish_gnn

python scripts/steering_perturbation_within.py --dataset aging_coronal --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/steering_perturbation_within.py --dataset aging_coronal --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/model_performance.py --dataset aging_sagittal --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/model_performance.py --dataset aging_sagittal --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/model_performance.py --dataset allen --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,Macrophage,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/model_performance.py --dataset allen --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,Macrophage,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/model_performance.py --dataset androvic --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,Macrophage,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/model_performance.py --dataset androvic --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,Macrophage,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/model_performance.py --dataset exercise --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/model_performance.py --dataset exercise --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/model_performance.py --dataset kukanja --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,Macrophage,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/model_performance.py --dataset kukanja --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,Macrophage,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/steering_perturbation_within.py --dataset liverperturb --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NK cell,B cell" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/steering_perturbation_within.py --dataset liverperturb --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NK cell,B cell" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/steering_perturbation_within.py --dataset lohoff --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "all" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/steering_perturbation_within.py --dataset lohoff --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "all" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/model_performance.py --dataset pilot --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/model_performance.py --dataset pilot --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "T cell,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/model_performance.py --dataset reprogramming --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "Macrophage,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/model_performance.py --dataset reprogramming --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "Macrophage,NSC,Pericyte" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

python scripts/model_performance.py --dataset zeng --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "VSMC,Microglia" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "test"
python scripts/model_performance.py --dataset zeng --base_path data/anndata --k_hop 2 --augment_hop 2 --center_celltypes "VSMC,Microglia" --node_feature "expression" --inject_feature "none" --learning_rate 0.0001 --loss "weightedl1" --epochs 50 --steering_approach "batch_steer_cell" --num_props 10 --train_or_test "train"

conda deactivate