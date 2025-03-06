#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J steercell # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 3600 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o steercell_%A.out # Standard out goes to this file 
#SBATCH -e steercell_%A.err # Standard err goes to this filehostname 

source activate merfish_gnn
python steering_perturbation.py batch_steer_cell 10 test none
#python steering_perturbation.py batch_steer_cell 10 train none
source deactivate