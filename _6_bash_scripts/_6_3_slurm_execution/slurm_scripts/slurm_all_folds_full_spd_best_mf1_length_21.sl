#!/bin/bash

# An example Slurm submission file to train the model on all folds, using existing hyperparameters
# Made for the French national supercomputer Jean Zay

# Account used (may be removed in some circumstances)
#SBATCH -A wpd@a100

# Job name
#SBATCH -J "all_folds_full_spd_best_mf1_length_21"

# Job output and error files
# Repeating the %a (array ID) at the beginning for better alphabetical file sorting
#SBATCH --output log/index_%a.job_%x.job_id_%j.master_id_%A.array_id_%a.out
#SBATCH --error log/index_%a.job_%x.job_id_%j.master_id_%A.array_id_%a.err

# Partition (submission class) - to adapt to your machine!
#SBATCH -C a100
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1

# Job time (hh:mm:ss)
#SBATCH --time 20:00:00

#SBATCH --mail-type ALL
# User e-mail address
# #SBATCH --mail-user your@address.here

# Running on all 31 folds
# Offset of 100, to run simultaneously with other jobs without conflict (important for Lightning logs!)
#SBATCH --array=100-130
#SBATCH --mail-type=ARRAY_TASKS

# Fold ID, removing the offset
fold=$((${SLURM_ARRAY_TASK_ID}-100))

# Loading execution environment
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/1.11.0
PATH=$PATH:~/.local/bin
export PATH

set -x
srun python -u command_line_runner.py --execution_method from_hparams --execution_type fit --global_seed 42 --trainer_config_file trainer_default_config.yaml --trainer_config.logger_version ${SLURM_ARRAY_TASK_ID} --hparams_config_file prevectorized_spd_network_length_21_best_mf1_hparams.yaml --datamodule_config.batch_size 64 --datamodule_config.cross_validation_fold_index $fold




