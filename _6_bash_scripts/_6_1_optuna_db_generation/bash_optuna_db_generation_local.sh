#!/bin/bash

# The database creation script, for local use

# Move to the script directory, then to the root, and create the folder "db"
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd ../..
mkdir -p db

# The following allows for the deletion and generation of Optuna studies within a single database file.
# Studies may be added to or removed from the file without affecting other studies already present.
# The direction should be set to "maximize" when tracking the accuracy, MF1 score, etc.
# and set to "minimize" when tracking the loss.

# optuna delete-study --study-name full_spd --storage sqlite:///db/database.db
optuna create-study --study-name full_spd --direction maximize --storage sqlite:///db/database.db


