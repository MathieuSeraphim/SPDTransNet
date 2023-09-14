import warnings
from os.path import realpath, dirname, join, isfile
from jsonargparse import ArgumentParser
from _2_data_preprocessing._2_3_preprocessors.BasePreprocessor import BasePreprocessor
from _2_data_preprocessing._2_3_preprocessors.ProcessorWrapper import PreprocessorWrapper


class BaseDataReader:

    COMPATIBLE_PREPROCESSOR_CLASS = BasePreprocessor

    def __init__(self, preprocessing_config_file: str, **kwargs):
        current_script_directory = dirname(realpath(__file__))
        root_directory = dirname(dirname(current_script_directory))
        configs_directory = join(root_directory, "_1_configs")
        preprocessing_configs_directory = join(configs_directory, "_1_1_preprocessing")
        full_preprocessing_config_file = join(preprocessing_configs_directory, preprocessing_config_file)

        assert isfile(full_preprocessing_config_file)
        assert full_preprocessing_config_file[-5:] == ".yaml"

        parser = ArgumentParser()
        parser.add_class_arguments(PreprocessorWrapper, "wrapper")
        preprocessor_config = parser.parse_path(full_preprocessing_config_file)
        self.corresponding_preprocessor_object = parser.instantiate_classes(preprocessor_config).wrapper.preprocessor
        assert isinstance(self.corresponding_preprocessor_object, self.COMPATIBLE_PREPROCESSOR_CLASS)

        if type(self) == BaseDataReader:
            warnings.warn("This is a base instance of a data reader, with no functionality.")
        else:
            self.parse_preprocessor_attributes()
            self.parse_initialization_arguments(**kwargs)

    def parse_preprocessor_attributes(self):
        raise NotImplementedError

    def parse_initialization_arguments(self, **kwargs):
        raise NotImplementedError

    def setup(self, **kwargs):
        raise NotImplementedError

    def get_element_data(self, element_id: int):
        raise NotImplementedError



