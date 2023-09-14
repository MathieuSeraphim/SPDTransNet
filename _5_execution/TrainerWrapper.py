from os.path import dirname, realpath, join, isfile
from typing import Dict, Union, Any
import yaml
from jsonargparse import ArgumentParser
from pytorch_lightning import Trainer


class TrainerWrapper:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer


def get_trainer_config_dict_from_file(config_file: str, logger_logs_location: str = ".",
                                      logger_logs_folder_name: str = "lightning_logs",
                                      logger_version: Union[int, None] = None):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(current_script_directory)
    configs_directory = join(root_directory, "_1_configs")
    trainer_configs_directory = join(configs_directory, "_1_6_trainer")
    trainer_config_file = join(trainer_configs_directory, config_file)
    assert isfile(trainer_config_file)

    trainer_config_dict = yaml.safe_load(open(trainer_config_file, "r"))
    trainer_config_dict["logger"][0]["init_args"]["save_dir"] = logger_logs_location
    trainer_config_dict["logger"][0]["init_args"]["name"] = logger_logs_folder_name
    trainer_config_dict["logger"][0]["init_args"]["version"] = logger_version

    return trainer_config_dict


def get_trainer_from_config_file(config_file: str, logger_logs_location: str = ".",
                                 logger_logs_folder_name: str = "lightning_logs",
                                 logger_version: Union[int, None] = None):
    trainer_config_dict = get_trainer_config_dict_from_file(config_file, logger_logs_location, logger_logs_folder_name,
                                                            logger_version)
    trainer = get_trainer_from_config_dict(trainer_config_dict)
    return trainer


def get_trainer_from_config_dict(config_dict: Dict[str, Any]):
    parser = ArgumentParser()

    wrapper_dict = {"wrapper":
        {"trainer":
            {
                "class_path": "pytorch_lightning.Trainer",
                "init_args": config_dict
            }
        }
    }

    parser.add_class_arguments(TrainerWrapper, "wrapper", fail_untyped=False)
    constructed_trainer = parser.instantiate_classes(wrapper_dict).wrapper.trainer
    return constructed_trainer

