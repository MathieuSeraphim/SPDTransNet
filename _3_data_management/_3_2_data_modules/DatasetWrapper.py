from torch.utils.data import Dataset
from os.path import realpath, dirname, join, isfile
from jsonargparse import ArgumentParser

# This wrapper is a botched solution; I couldn't figure out how to do a YAML config import through jsonargparse the way
# I wanted without it.


class DatasetWrapper:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset


def get_dataset_from_config(config_file: str):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(current_script_directory))
    configs_directory = join(root_directory, "_1_configs")
    dataset_configs_directory = join(configs_directory, "_1_3_datasets")
    dataset_config_file = join(dataset_configs_directory, config_file)
    assert isfile(dataset_config_file)

    parser = ArgumentParser()
    parser.add_class_arguments(DatasetWrapper, "wrapper")
    data_reader_config = parser.parse_path(dataset_config_file)
    constructed_dataset = parser.instantiate_classes(data_reader_config).wrapper.dataset
    return constructed_dataset
