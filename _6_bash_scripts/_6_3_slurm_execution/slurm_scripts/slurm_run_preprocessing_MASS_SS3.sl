#!/bin/bash

# An example Slurm submission file to apply the preprocessing to the extracted MASS SS3 dataset
# Made for the French national supercomputer Jean Zay

# Account used (may be removed in some circumstances)
#SBATCH -A wpd@v100

# Job name
#SBATCH -J "preprocessing"

# Job output and error files
# Repeating the %a (array ID) at the beginning for better alphabetical file sorting
#SBATCH --output log/index_%a.job_%x.job_id_%j.master_id_%A.array_id_%a.out
#SBATCH --error log/index_%a.job_%x.job_id_%j.master_id_%A.array_id_%a.err

# Partition (submission class) - to adapt to your machine!
#SBATCH --cpus-per-task=10
#SBATCH --tasks-per-node=1
#SBATCH --gres gpu:1

# Job time (hh:mm:ss)
#SBATCH --time 10:00:00

#SBATCH --mail-type ALL
# User e-mail address
# #SBATCH --mail-user your@address.here

# Single job, choose an ID that doesn't clash with running jobs
#SBATCH --array=2048
#SBATCH --mail-type=ARRAY_TASKS

# Loading execution environment
module purge
module load pytorch-gpu/py3/1.11.0
PATH=$PATH:~/.local/bin
export PATH

set -x
srun python -u run_preprocessing.py SPD_matrices_from_EEG_MASS_SS3_dataset_EUSIPCO_signals_config.yaml



