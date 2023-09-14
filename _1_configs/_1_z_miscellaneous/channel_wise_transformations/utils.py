from os.path import dirname, realpath, join, isfile
from typing import List

import yaml


def get_channel_wise_transformations(config_file: str, list_of_transformations: List[str]):
    current_script_directory = dirname(realpath(__file__))
    transformations_config_file = join(current_script_directory, config_file)
    assert isfile(transformations_config_file)
    transformations_dict = yaml.safe_load(open(transformations_config_file, "r"))["channel_wise_transformations"]

    output_list = []
    for transformation in list_of_transformations:
        assert transformation in transformations_dict
        output_list.append(transformations_dict[transformation])

    return output_list

