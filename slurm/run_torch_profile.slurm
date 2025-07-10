#!/bin/bash
#SBATCH -J k2_wL14_Ecpu # job name
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2880 # Runtime in minutes 
#SBATCH -p jamesz # Partition to submit to 
#SBATCH --account=jamesz
#SBATCH --nodelist=sh03-10n67,sh03-10n68
#SBATCH --mem=100G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o k2_wL14_Ecpu_%A.out # Standard out goes to this file 
#SBATCH -e k2_wL14_Ecpu_%A.err # Standard err goes to this filehostname 

source activate merfish_gnn
python -m torch.utils.bottleneck train_gnn_model_expression.py 2 2 "T cell,NSC,Pericyte" "expression" "none" 0.0001 "weightedl1"
source deactivate