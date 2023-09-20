#!/bin/bash

SCRIPT_DIRECTORY=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
ROOT_DIRECTORY=$(dirname "$(dirname "$SCRIPT_DIRECTORY")")
SLURM_SCRIPTS_DIRECTORY=$SCRIPT_DIRECTORY/slurm_scripts

slurm_default_script=$SLURM_SCRIPTS_DIRECTORY/_slurm_default.sl
slurm_script=${1:-$slurm_default_script}

cd "$ROOT_DIRECTORY"
mkdir -p log
sbatch "$SLURM_SCRIPTS_DIRECTORY"/"$slurm_script"