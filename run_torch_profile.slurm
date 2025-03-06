#!/bin/bash
#SBATCH -n 8 # Number of cores requested 
#SBATCH -J torch_profile # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 3600 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=64G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o torch_profile%A.out # Standard out goes to this file 
#SBATCH -e torch_profile%A.err # Standard err goes to this filehostname 

source activate merfish_gnn
python -m torch.utils.bottleneck train_gnn_model_expression.py 2 2 "T cell,NSC,Pericyte" "expression" "none" 0.0001 "weightedl1"
source deactivate