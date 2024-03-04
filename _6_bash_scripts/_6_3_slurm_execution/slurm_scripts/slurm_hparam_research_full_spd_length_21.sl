#!/bin/bash

# An example Slurm submission file to run a hyperparameter research
# Made for the French national supercomputer Jean Zay

# Account used (may be removed in some circumstances)
#SBATCH -A wpd@a100

# Job name
#SBATCH -J "hparam_research_full_spd_length_21"

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
#SBATCH --time 10:00:00

#SBATCH --mail-type ALL
# User e-mail address
# #SBATCH --mail-user your@address.here

# Research over 50 jobs, with max. 5 concurrent jobs
#SBATCH --array=0-49%5
#SBATCH --mail-type=ARRAY_TASKS

# Loading execution environment
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/1.11.0
PATH=$PATH:~/.local/bin
export PATH

# The fold index is set to 11 for hyperparameter researches (with zero-indexing)
# Initially a random choice, now preserved for the sake of consistency
set -x
srun python -u command_line_runner.py --optuna_flag --optuna.study_name full_spd --optuna.hparam_selection_config.model SPD_to_EEG_spd_preserving_prevectorized_config.yaml --optuna.hparam_selection_config.datamodule Vectorized_SPD_from_EEG_config.yaml --optuna.pruner.n_startup_trials 5 --execution_method standalone --execution_type fit --global_seed 42 --trainer_config_file trainer_default_config.yaml --trainer_config.logger_version ${SLURM_ARRAY_TASK_ID} --model_config_file EUSIPCO_signals_spd_preserving_network_prevectorized_length_21_config.yaml --datamodule_config_file Vectorized_SPD_matrices_from_EEG_MASS_dataset_EUSIPCO_signals_length_21_config.yaml --datamodule_config.batch_size 32 --datamodule_config.cross_validation_fold_index 11




