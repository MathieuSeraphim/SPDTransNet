#!/bin/bash

# An example BASH script to train our model with existing hyperparameters on a single fold
# All other Slurm scripts may be adapted in this way for local use
# As all our runs were done through Slurm, we gave not developed scripts for local sequential or multi-GPU usage

# The fold index will be either the first passed argument, or 11
fold=${1:-11}

# Move to the script directory, then to the root
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd ../..

python command_line_runner.py --execution_method from_hparams --execution_type train --global_seed 42 --trainer_config_file trainer_default_config.yaml --trainer_config.logger_version $fold --hparams_config_file prevectorized_spd_network_length_21_best_mf1_hparams.yaml --datamodule_config.batch_size 64 --datamodule_config.cross_validation_fold_index $fold




