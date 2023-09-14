from collections import Sequence
from os.path import join, isfile
from typing import Dict, Any, Union
import yaml
from optuna import Trial
from _5_execution.utils import get_execution_configs_folder_path


def optuna_suggest_hparam(trial: Trial, hparam_name: str, suggestion_type: str, suggestion_arguments: Any):
    if suggestion_type == "categorical":
        assert isinstance(suggestion_arguments, Sequence)
        return trial.suggest_categorical(hparam_name, suggestion_arguments)

    if suggestion_type == "int":
        assert len(suggestion_arguments) == 2
        lower, upper = suggestion_arguments
        lower_int = int(lower)
        upper_int = int(upper)
        assert (lower_int, upper_int) == (lower, upper)
        assert lower_int < upper_int
        return trial.suggest_int(hparam_name, lower_int, upper_int)

    elif suggestion_type == "loguniform":
        assert len(suggestion_arguments) == 2
        lower, upper = suggestion_arguments
        lower_float = float(lower)
        upper_float = float(upper)
        assert 0 < lower_float < upper_float
        return trial.suggest_loguniform(hparam_name, lower_float, upper_float)

    raise NotImplementedError


suggestion_types_list = ["categorical", "loguniform", "int"]
nested_identifier = "NESTED"
complex_identifier = "COMPLEX"
conditional_identifier = "CONDITIONAL"
special_identifiers = [nested_identifier, complex_identifier, conditional_identifier]
def optuna_suggest_hparams(trial: Trial, class_init_args_dict: Dict[str, Any], hparams_choice_dict: Dict[str, Any]):
    hparam_keys = hparams_choice_dict.keys()

    for hparam_key in hparam_keys:

        if hparam_key not in special_identifiers:
            hparam_dict = hparams_choice_dict[hparam_key]
            hparam_name = hparam_key
            assert hparam_name in class_init_args_dict.keys()

            suggestion_type = hparam_dict["type"]
            assert suggestion_type in suggestion_types_list
            suggestion_arguments = hparam_dict["args"]
            class_init_args_dict[hparam_name] = optuna_suggest_hparam(trial, hparam_name, suggestion_type, suggestion_arguments)
        
        else:

            # Regular values, but nested in sub-dicts of the config file
            if hparam_key == nested_identifier:
                nested_hparams_list = hparams_choice_dict[hparam_key]

                for nested_hparams_list_item in nested_hparams_list:
                    nested_hparam_name = nested_hparams_list_item["item"]
                    base_nested_path = nested_hparams_list_item["path"]
                    full_nested_hparam_name = base_nested_path + "/" + nested_hparam_name

                    nested_hparam_path = base_nested_path.split("/")
                    
                    suggestion_type = nested_hparams_list_item["type"]
                    assert suggestion_type in suggestion_types_list
                    suggestion_arguments = nested_hparams_list_item["args"]

                    nested_init_args_dict = class_init_args_dict
                    for key in nested_hparam_path:
                        nested_init_args_dict = nested_init_args_dict[key]

                    assert nested_hparam_name in nested_init_args_dict.keys()
                    nested_init_args_dict[nested_hparam_name] = optuna_suggest_hparam(trial, full_nested_hparam_name, suggestion_type, suggestion_arguments)

            # Values other than None, bool, int, float or str - treated as categorical
            elif hparam_key == complex_identifier:
                complex_hparams_list = hparams_choice_dict[hparam_key]

                for complex_hparams_list_item in complex_hparams_list:
                    complex_hparam_name = complex_hparams_list_item["item"]
                    base_nested_path_if_any = complex_hparams_list_item["path_if_nested"]
                    full_complex_hparam_name = complex_hparam_name
                    if base_nested_path_if_any is not None:
                        full_complex_hparam_name = base_nested_path_if_any + "/" + complex_hparam_name
                    
                    suggested_values_list = complex_hparams_list_item["args"]
                    corresponding_categories_list = list(range(len(suggested_values_list)))
                    
                    suggested_category_index = optuna_suggest_hparam(trial, full_complex_hparam_name, "categorical", corresponding_categories_list)

                    potentially_nested_init_args_dict = class_init_args_dict
                    if base_nested_path_if_any is not None:
                        nested_hparam_path = base_nested_path_if_any.split("/")
                        for key in nested_hparam_path:
                            potentially_nested_init_args_dict = potentially_nested_init_args_dict[key]
                    
                    potentially_nested_init_args_dict[complex_hparam_name] = suggested_values_list[suggested_category_index]
                    
            # Potentially nested simple value, that is conditional on some other hyperparameter being set to a specific value
            elif hparam_key == conditional_identifier:
                conditional_hparams_list = hparams_choice_dict[hparam_key]
                
                for conditional_hparams_list_item in conditional_hparams_list:
                    condition_value = conditional_hparams_list_item["condition_value"]
                    condition_not_met_flag = False
                    
                    condition_dict_or_item = class_init_args_dict
                    potentially_nested_condition_path = conditional_hparams_list_item["path_to_condition"].split("/")
                    for key in potentially_nested_condition_path:
                        if key not in condition_dict_or_item.keys():
                            condition_not_met_flag = True
                            break
                        condition_dict_or_item = condition_dict_or_item[key]
                    
                    if condition_not_met_flag:
                        continue
                    if condition_dict_or_item != condition_value:
                        continue

                    conditional_hparam_name = conditional_hparams_list_item["item"]
                    base_nested_path_if_any = conditional_hparams_list_item["path_if_nested"]
                    full_conditional_hparam_name = conditional_hparam_name
                    if base_nested_path_if_any is not None:
                        full_conditional_hparam_name = base_nested_path_if_any + "/" + conditional_hparam_name

                    suggestion_type = conditional_hparams_list_item["type"]
                    assert suggestion_type in suggestion_types_list or suggestion_type is None
                    suggestion_arguments = conditional_hparams_list_item["args"]

                    potentially_nested_init_args_dict = class_init_args_dict
                    if base_nested_path_if_any is not None:
                        nested_hparam_path = base_nested_path_if_any.split("/")
                        for key in nested_hparam_path:
                            potentially_nested_init_args_dict = potentially_nested_init_args_dict[key]

                    if suggestion_type is not None:
                        potentially_nested_init_args_dict[conditional_hparam_name] = optuna_suggest_hparam(trial, full_conditional_hparam_name, suggestion_type, suggestion_arguments)
                    else:
                        assert len(suggestion_arguments) == 1
                        potentially_nested_init_args_dict[conditional_hparam_name] = suggestion_arguments[0]

    return class_init_args_dict


def optuna_hparam_selection(trial: Trial, class_dict: Dict[str, Any], hparams_choice_dict: Dict[str, Any]):
    init_args = class_dict["init_args"]

    init_args = optuna_suggest_hparams(trial, init_args, hparams_choice_dict)
    class_dict["init_args"] = init_args

    return class_dict


class_types = ["model", "data_module"]
def optuna_hparam_selection_from_file(trial: Trial, class_type: str, class_dict: Dict[str, Any],
                                      hparams_choice_file: Union[str, None]):
    if hparams_choice_file is not None:
        selection_configs_folder = get_execution_configs_folder_path("hyperparameter_selection")
        assert class_type in class_types
        hparams_choice_filename = join(join(selection_configs_folder, class_type + "s"), hparams_choice_file)
        assert isfile(hparams_choice_filename)

        hparams_choice_dict = yaml.safe_load(open(hparams_choice_filename, "r"))

    else:
        hparams_choice_dict = {}

    return optuna_hparam_selection(trial, class_dict, hparams_choice_dict)

