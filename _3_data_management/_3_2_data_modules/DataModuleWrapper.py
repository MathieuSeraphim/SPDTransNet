from typing import Union, Dict, Any

import yaml
from jsonargparse import ArgumentParser
from _3_data_management._3_2_data_modules.BaseDataModule import BaseDataModule
from os.path import realpath, dirname, join, isfile


# This wrapper is a botched solution; I couldn't figure out how to do a YAML config import through jsonargparse the way
# I wanted without it.


class DataModuleWrapper:

    def __init__(self, data_module: BaseDataModule):
        self.data_module = data_module


def get_datamodule_config_dict_from_file(config_file: str):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(current_script_directory))
    configs_directory = join(root_directory, "_1_configs")
    data_modules_configs_directory = join(configs_directory, "_1_4_data_modules")
    data_module_config_file = join(data_modules_configs_directory, config_file)
    assert isfile(data_module_config_file)

    return yaml.safe_load(open(data_module_config_file, "r"))


def get_datamodule_from_config_file(config_file: str, batch_size: Union[int, None] = 64,
                                    cross_validation_fold_index: Union[int, None] = 11):

    data_module_config_dict = get_datamodule_config_dict_from_file(config_file)
    data_module_config_dict = modify_datamodule_config_dict(data_module_config_dict, batch_size,
                                                            cross_validation_fold_index)
    return get_datamodule_from_config_dict(data_module_config_dict)


def modify_datamodule_config_dict(config_dict: Dict[str, Any], batch_size: Union[int, None] = 64,
                                  cross_validation_fold_index: Union[int, None] = 11):
    if batch_size is not None:
        config_dict["init_args"]["batch_size"] = batch_size
    if cross_validation_fold_index is not None:
        config_dict["init_args"]["cross_validation_fold_index"] = cross_validation_fold_index
    return config_dict


def get_datamodule_from_config_dict(config_dict: Dict[str, Any]):
    parser = ArgumentParser()
    wrapper_dict = {"wrapper": {"data_module": config_dict}}
    parser.add_class_arguments(DataModuleWrapper, "wrapper", fail_untyped=False)
    constructed_data_module = parser.instantiate_classes(wrapper_dict).wrapper.data_module
    return constructed_data_module


if __name__ == "__main__":
    test_dict = {
        "wrapper": {
            "data_module": {
                "class_path": "_3_data_management._3_2_data_modules.BaseDataModule.BaseDataModule",
                "init_args": {
                    "batch_size": 64,
                    "dataloader_num_workers": 0,
                    "random_seed": 42
                }
            }
        }
    }

    parser = ArgumentParser()
    parser.add_class_arguments(DataModuleWrapper, "wrapper", fail_untyped=False)
    constructed_data_module = parser.instantiate_classes(test_dict).wrapper.data_module
    print("Loaded data module of class:", constructed_data_module.__class__.__name__)
