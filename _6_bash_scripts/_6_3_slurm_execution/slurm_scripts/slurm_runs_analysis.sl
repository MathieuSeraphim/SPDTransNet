#!/bin/bash

# An example Slurm submission file to generate the test set results on all folds using trained model weights
# Made for the French national supercomputer Jean Zay

# Account used (may be removed in some circumstances)
#SBATCH -A wpd@v100

# Job name
#SBATCH -J "runs_analysis"

# Batch output file
#SBATCH --output log/index_%a.job_%x.job_id_%j.master_id_%A.array_id_%a.out

# Batch error file
#SBATCH --error log/index_%a.job_%x.job_id_%j.master_id_%A.array_id_%a.err

# Partition (submission class) - to adapt to your machine!
#SBATCH --qos qos_gpu-t3
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1

# Job time (hh:mm:ss)
#SBATCH --time 20:00:00

#SBATCH --mail-type ALL
# User e-mail address
# #SBATCH --mail-user your@address.here

# Single job, choose an ID that doesn't clash with running jobs
#SBATCH --array=4096
#SBATCH --mail-type=ARRAY_TASKS

# Loading execution environment
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/1.11.0
PATH=$PATH:~/.local/bin
export PATH

# By default, will analyze all entries in the "lightning_logs" folder, generated at the project root
set -x
srun python -u command_line_tester.py --run_analysis_config_name _SPD_from_EEG_base_runs_analysis_config.yaml




