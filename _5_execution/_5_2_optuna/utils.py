from typing import Dict, Any
from optuna.trial import Trial
from pytorch_lightning.callbacks import Callback
from _3_data_management._3_2_data_modules.DataModuleWrapper import get_datamodule_config_dict_from_file, \
    modify_datamodule_config_dict
from _4_models.ModelWrapper import get_model_config_dict_from_file, modify_model_config_dict
from _5_execution.TrainerWrapper import get_trainer_config_dict_from_file
from _5_execution._5_2_optuna.hparam_selection import optuna_hparam_selection_from_file
from _5_execution.run_model import get_model_and_datamodule_dicts_from_hparams_file
from jsonargparse import Namespace


def optuna_standalone_execution(trainer_config_file: str, trainer_config_kwargs: Dict[str, str], model_config_file: str,
                                model_config_kwargs: Dict[str, str], datamodule_config_file: str,
                                datamodule_config_kwargs: Dict[str, str]):
    trainer_config_dict = get_trainer_config_dict_from_file(trainer_config_file, **trainer_config_kwargs)
    model_config_dict = get_model_config_dict_from_file(model_config_file)
    model_config_dict = modify_model_config_dict(model_config_dict, **model_config_kwargs)
    datamodule_config_dict = get_datamodule_config_dict_from_file(datamodule_config_file)
    datamodule_config_dict = modify_datamodule_config_dict(datamodule_config_dict, **datamodule_config_kwargs)

    return trainer_config_dict, model_config_dict, datamodule_config_dict


def optuna_execution_on_previously_obtained_hparams(trainer_config_file: str, trainer_config_kwargs: Dict[str, str],
                                                    model_config_kwargs: Dict[str, str],
                                                    datamodule_config_kwargs: Dict[str, str], hparams_file: str):
    trainer_config_dict = get_trainer_config_dict_from_file(trainer_config_file, **trainer_config_kwargs)
    model_config_dict, datamodule_config_dict = get_model_and_datamodule_dicts_from_hparams_file(hparams_file)
    model_config_dict = modify_model_config_dict(model_config_dict, **model_config_kwargs)
    datamodule_config_dict = modify_datamodule_config_dict(datamodule_config_dict, **datamodule_config_kwargs)

    return trainer_config_dict, model_config_dict, datamodule_config_dict


def before_class_instantiation(trial: Trial, model_dict: Dict[str, Any], datamodule_dict: Dict[str, Any],
                               hparam_selection_files: Namespace):
    return optuna_hparam_selection_from_file(trial, "model", model_dict, hparam_selection_files.model),\
           optuna_hparam_selection_from_file(trial, "data_module", datamodule_dict, hparam_selection_files.datamodule)


def add_callback_to_trainer_config_dict(trainer_config_dict: Dict[str, Any], callback: Callback):
    if "callbacks" not in trainer_config_dict.keys():
        trainer_config_dict["callbacks"] = []
    trainer_config_dict["callbacks"].append(callback)
    return trainer_config_dict






