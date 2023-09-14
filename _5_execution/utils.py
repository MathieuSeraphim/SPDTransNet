from os.path import join, dirname, realpath
from pathlib import Path
from typing import Any, cast, Dict
from lightning_fabric.utilities.cloud_io import _load as pl_load
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE
from lightning_utilities.core.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.migration.utils import _pl_migrate_checkpoint
from _4_models.ModelWrapper import get_model_from_config_dict


def get_execution_configs_folder_path(folder: str):
    current_script_directory = dirname(realpath(__file__))
    miscellaneous_directory = join(current_script_directory, "_5_z_configs")
    return join(miscellaneous_directory, folder)


# The model we're using needs to call the setup method at least once before loading checkpoints.
# It also can't be initiated by standard Lightning functions using a .yaml config file because of the non-standard way
# we are dealing with those.
# Hence the following functions.
###################################################################################################################


# Adapted from pytorch_lightning/core/saving.py
def preparing_checkpoint_object(checkpoint_file: str, model_class: Any):
    map_location = cast(_MAP_LOCATION_TYPE, lambda storage, loc: storage)
    with pl_legacy_patch():
        checkpoint = pl_load(checkpoint_file, map_location=map_location)
    checkpoint = _pl_migrate_checkpoint(
        checkpoint, checkpoint_path=(checkpoint_file if isinstance(checkpoint_file, (str, Path)) else None)
    )
    checkpoint.setdefault(model_class.CHECKPOINT_HYPER_PARAMS_KEY, {})
    return checkpoint


# Adapted from pytorch_lightning/core/saving.py
def handle_keys_warning(keys: Any):
    if keys.missing_keys:
        rank_zero_warn(
            f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
        )
    if keys.unexpected_keys:
        rank_zero_warn(
            f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
        )


def get_model_from_checkpoint_file(model_dict: Dict[str, Any], checkpoint_file: str):
    model = get_model_from_config_dict(model_dict)
    model.setup(None)
    model_class = model.__class__
    checkpoint_dict = preparing_checkpoint_object(checkpoint_file, model_class)
    model.on_load_checkpoint(checkpoint_dict)
    keys = model.load_state_dict(checkpoint_dict["state_dict"], strict=False)
    handle_keys_warning(keys)
    return model

