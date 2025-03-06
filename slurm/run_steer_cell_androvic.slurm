#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J steerinjury # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 3600 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o steerinjury_%A.out # Standard out goes to this file 
#SBATCH -e steerinjury_%A.err # Standard err goes to this filehostname 

source activate merfish_gnn
python steering_perturbation.py batch_steer_cell 10 test injury
python steering_perturbation.py batch_steer_cell 10 test noninjury
source deactivate