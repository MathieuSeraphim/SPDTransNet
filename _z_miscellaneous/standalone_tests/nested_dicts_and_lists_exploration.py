import json
import pickle
import sys
from typing import Any, Union, List, Dict, Sequence
import numpy as np
import torch


def item_to_string(item: Any):
    if isinstance(item, str):
        return item
    if isinstance(item, int) or isinstance(item, float):
        return str(item)

    item_as_string = str(item.__class__)
    assert item_as_string[0] == "<"
    assert item_as_string[-1] == ">"

    if isinstance(item, np.ndarray):
        shape_tuple = item.shape
    elif isinstance(item, torch.Tensor):
        shape_tuple = tuple([int(s) for s in item.shape])
    else:
        return item_as_string

    return "%s, shape: %s>" % (item_as_string[:-1], str(shape_tuple))


def recursive_exploration(item: Any):
    if isinstance(item, dict):
        output_dict = {}
        for key, value in item.items():
            output_dict[key] = recursive_exploration(value)
        return output_dict

    if isinstance(item, list) or isinstance(item, tuple):
        output_list = []
        for element in item:
            output_list.append(recursive_exploration(element))
        return output_list

    return item_to_string(item)


def recursive_exploration_from_pickle_file(pickle_file_full_path: str):
    with open(pickle_file_full_path, "rb") as f:
        structure_to_explore = pickle.load(f)
    return recursive_exploration(structure_to_explore)


def pretty_print_recursive_exploration(recursive_exploration_output: Union[List, Dict, str], indent: int = 2):
    print(json.dumps(recursive_exploration_output, indent=indent))


if __name__ == "__main__":
    # pass
    filename = "your_file_path_here.pkl"  # Either an absolute path, or a path relative to the location of execution.
    pretty_print_recursive_exploration(recursive_exploration_from_pickle_file(filename))
