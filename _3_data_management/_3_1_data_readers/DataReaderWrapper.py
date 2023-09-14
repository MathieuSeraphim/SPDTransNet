from _3_data_management._3_1_data_readers.BaseDataReader import BaseDataReader
from os.path import realpath, dirname, join, isfile
from jsonargparse import ArgumentParser

# This wrapper is a botched solution; I couldn't figure out how to do a YAML config import through jsonargparse the way
# I wanted without it.


class DataReaderWrapper:

    def __init__(self, data_reader: BaseDataReader):
        self.data_reader = data_reader


def get_data_reader_from_config(config_file: str):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(current_script_directory))
    configs_directory = join(root_directory, "_1_configs")
    data_reading_configs_directory = join(configs_directory, "_1_2_data_reading")
    data_reader_config_file = join(data_reading_configs_directory, config_file)
    assert isfile(data_reader_config_file)

    parser = ArgumentParser()
    parser.add_class_arguments(DataReaderWrapper, "wrapper")
    data_reader_config = parser.parse_path(data_reader_config_file)
    constructed_data_reader = parser.instantiate_classes(data_reader_config).wrapper.data_reader
    return constructed_data_reader


if __name__ == "__main__":
    data_reader = get_data_reader_from_config("_data_reading_config_template.yaml")
    print("Loaded data reader of class:", data_reader.__class__.__name__)
