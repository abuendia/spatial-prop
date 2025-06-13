#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J s_exercise # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 500 # Runtime in minutes 
#SBATCH --partition=jamesz # Partition to submit to 
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o stdout/spatial_embeds_exercise.out # Standard out goes to this file 
#SBATCH -e stderr/spatial_embeds_exercise.err # Standard err goes to this filehostname 

source ~/.bashrc
conda activate spatial
python /oak/stanford/groups/jamesz/abuen/spatial-rotation/repos/spatial-gnn/spatial-genept/geneptw_baseline.py --adata_path /oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/raw/exercise.h5ad --output_path /oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/exercise_embeds_all_geneptw.json
