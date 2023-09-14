from os.path import join, isfile
import yaml
from pytorch_lightning import Trainer, seed_everything
from typing import Dict, Union
from _3_data_management._3_2_data_modules.BaseDataModule import BaseDataModule
from _3_data_management._3_2_data_modules.DataModuleWrapper import get_datamodule_from_config_file, \
    get_datamodule_from_config_dict, modify_datamodule_config_dict
from _4_models.BaseModel import BaseModel
from _4_models.ModelWrapper import get_model_from_config_file, get_model_from_config_dict, modify_model_config_dict
from _5_execution.TrainerWrapper import get_trainer_from_config_file
from _5_execution.utils import get_execution_configs_folder_path

trainer_modes = ["fit", "validate", "test", "predict"]


def run_model(trainer_mode: str, trainer: Trainer, model: BaseModel, datamodule: BaseDataModule,
              random_seed: Union[int, None] = 42):
    assert trainer_mode in trainer_modes

    print()
    print("All classes initialized. Running model.")
    print("Execution mode: %s" % trainer_mode)
    print("Model object class: %s" % model.__class__.__name__)
    print("Number of model parameters: %d" % sum(p.numel() for p in model.parameters()))
    print("Data module object class: %s" % datamodule.__class__.__name__)

    if random_seed is not None:
        seed_everything(random_seed, workers=True)

    debug_mode_string = "Debug mode "
    if __debug__:
        debug_mode_string += "enabled. Asserts ran."
    else:
        assert False  # Cheeky check
        debug_mode_string += "disabled. Asserts ignored."
    print(debug_mode_string)
    print()

    if trainer_mode == "fit":
        return trainer.fit(model, datamodule)
    if trainer_mode == "validate":
        return trainer.validate(model, datamodule)
    if trainer_mode == "test":
        return trainer.test(model, datamodule)
    if trainer_mode == "predict":
        return trainer.predict(model, datamodule)
    raise NotImplementedError


def standalone_execution(trainer_mode: str, trainer_config_file: str, trainer_config_kwargs: Dict[str, str],
                         model_config_file: str, model_config_kwargs: Dict[str, str],
                         datamodule_config_file: str, datamodule_config_kwargs: Dict[str, str],
                         random_seed: Union[int, None] = 42):

    trainer = get_trainer_from_config_file(trainer_config_file, **trainer_config_kwargs)
    model = get_model_from_config_file(model_config_file, **model_config_kwargs)
    datamodule = get_datamodule_from_config_file(datamodule_config_file, **datamodule_config_kwargs)

    return run_model(trainer_mode, trainer, model, datamodule, random_seed)


def get_model_and_datamodule_dicts_from_hparams_file(hparams_file: str, is_full_path: bool = False):
    if is_full_path:
        model_and_datamodule_config_file = hparams_file
    else:
        past_runs_hparams_folder = get_execution_configs_folder_path("past_runs_hyperparameters")
        model_and_datamodule_config_file = join(past_runs_hparams_folder, hparams_file)
    assert isfile(model_and_datamodule_config_file)

    config_dict = yaml.safe_load(open(model_and_datamodule_config_file, "r"))
    model_config_dict = config_dict["model"]
    datamodule_config_dict = config_dict["datamodule"]
    return model_config_dict, datamodule_config_dict


def get_model_and_datamodule_from_hparams_file(hparams_file: str, model_config_kwargs, datamodule_config_kwargs):
    model_config_dict, datamodule_config_dict = get_model_and_datamodule_dicts_from_hparams_file(hparams_file)

    model_config_dict = modify_model_config_dict(model_config_dict, **model_config_kwargs)
    model = get_model_from_config_dict(model_config_dict)

    datamodule_config_dict = modify_datamodule_config_dict(datamodule_config_dict, **datamodule_config_kwargs)
    datamodule = get_datamodule_from_config_dict(datamodule_config_dict)

    return model, datamodule


def execution_on_previously_obtained_hparams(trainer_mode: str, trainer_config_file: str,
                                             trainer_config_kwargs: Dict[str, str],
                                             model_config_kwargs: Dict[str, str],
                                             datamodule_config_kwargs: Dict[str, str], hparams_file: str,
                                             random_seed: Union[int, None] = 42):

    trainer = get_trainer_from_config_file(trainer_config_file, **trainer_config_kwargs)
    model, datamodule = get_model_and_datamodule_from_hparams_file(hparams_file, model_config_kwargs,
                                                                   datamodule_config_kwargs)

    return run_model(trainer_mode, trainer, model, datamodule, random_seed)



