from _2_data_preprocessing._2_3_preprocessors.BasePreprocessor import BasePreprocessor
from os.path import realpath, dirname, join, isfile
from jsonargparse import ArgumentParser

# This wrapper is a botched solution; I couldn't figure out how to do a YAML config import through jsonargparse the way
# I wanted without it.


class PreprocessorWrapper:

    def __init__(self, preprocessor: BasePreprocessor):
        self.preprocessor = preprocessor


def get_preprocessor_from_config(config_file: str):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(dirname(current_script_directory))
    configs_directory = join(root_directory, "_1_configs")
    preprocessing_configs_directory = join(configs_directory, "_1_1_preprocessing")
    preprocessor_config_file = join(preprocessing_configs_directory, config_file)
    assert isfile(preprocessor_config_file)

    parser = ArgumentParser()
    parser.add_class_arguments(PreprocessorWrapper, "wrapper")
    preprocessor_config = parser.parse_path(preprocessor_config_file)
    constructed_preprocessor = parser.instantiate_classes(preprocessor_config).wrapper.preprocessor
    return constructed_preprocessor


if __name__ == "__main__":
    preprocessor = get_preprocessor_from_config("_preprocessing_config_template.yaml")
    print("Loaded preprocessor of class:", preprocessor.__class__.__name__)