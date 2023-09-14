from typing import Any, Dict, Union
import yaml
from jsonargparse import ArgumentParser
from os.path import realpath, dirname, join, isfile
from _4_models.BaseModel import BaseModel


# This wrapper is a botched solution; I couldn't figure out how to do a YAML config import through jsonargparse the way
# I wanted without it.


class ModelWrapper:

    def __init__(self, model: BaseModel):
        self.model = model


def get_model_config_dict_from_file(config_file: str):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(current_script_directory)
    configs_directory = join(root_directory, "_1_configs")
    models_configs_directory = join(configs_directory, "_1_5_models")
    model_config_file = join(models_configs_directory, config_file)
    assert isfile(model_config_file)

    model_config_dict = yaml.safe_load(open(model_config_file, "r"))
    return model_config_dict


def get_model_from_config_file(config_file: str, cross_validation_fold_index: Union[int, None] = None):
    model_config_dict = get_model_config_dict_from_file(config_file)
    model_config_dict = modify_model_config_dict(model_config_dict, cross_validation_fold_index)
    return get_model_from_config_dict(model_config_dict)


def modify_model_config_dict(config_dict: Dict[str, Any], cross_validation_fold_index: Union[int, None] = None):
    if cross_validation_fold_index is not None:
        config_dict["init_args"]["fold_index"] = cross_validation_fold_index
    return config_dict


def get_model_from_config_dict(config_dict: Dict[str, Any]):
    parser = ArgumentParser()
    wrapper_dict = {"wrapper": {"model": config_dict}}
    parser.add_class_arguments(ModelWrapper, "wrapper", fail_untyped=False)
    constructed_model = parser.instantiate_classes(wrapper_dict).wrapper.model
    return constructed_model


if __name__ == "__main__":
    model = get_model_from_config_file("_base_model_config.yaml")
    print("Loaded first model of class:", model.__class__.__name__)
    sequence_to_classif_model = get_model_from_config_file("_sequence_to_classification_base_model_config.yaml")
    print("Loaded second model of class:", sequence_to_classif_model.__class__.__name__)
