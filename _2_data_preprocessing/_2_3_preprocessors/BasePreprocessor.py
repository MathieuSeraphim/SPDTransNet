import warnings
from os import mkdir, listdir
from os.path import realpath, dirname, join, isdir


class BasePreprocessor:

    def __init__(self, dataset_name: str, transformation_configuration_name: str, **kwargs):
        self.dataset_name = dataset_name
        self.transformation_config_name = transformation_configuration_name

        preprocessors_directory = dirname(realpath(__file__))
        data_preprocessing_directory = dirname(preprocessors_directory)
        data_extraction_directory = join(data_preprocessing_directory, "_2_2_data_extraction")
        self.extracted_dataset_folder = join(data_extraction_directory, dataset_name + "_extracted")

        self.preprocessed_data_directory = join(data_preprocessing_directory, "_2_4_preprocessed_data")

        self.preprocessed_data_folder = join(self.preprocessed_data_directory, "%s_dataset_with_%s_config"
                                             % (self.dataset_name, transformation_configuration_name))

        self.list_of_recording_pickle_filepaths = None

        if type(self) == BasePreprocessor:
            warnings.warn("This is a base instance of a preprocessor, with no functionality.")
        else:
            self.parse_initialization_arguments(**kwargs)

    def parse_initialization_arguments(self, **kwargs):
        raise NotImplementedError

    def preprocess(self):
        if not isdir(self.extracted_dataset_folder):
            raise FileNotFoundError("No extracted dataset folder was found for the \"%s\" dataset!" % self.dataset_name)
        if not isdir(self.preprocessed_data_directory):
            mkdir(self.preprocessed_data_directory)
        if not isdir(self.preprocessed_data_folder):
            mkdir(self.preprocessed_data_folder)
        self.list_of_recording_pickle_filepaths = [join(self.extracted_dataset_folder, filepath) for filepath in
                                                   listdir(self.extracted_dataset_folder) if filepath[-4:] == ".pkl"]
        self.list_of_recording_pickle_filepaths.sort()

        if type(self) == BasePreprocessor:
            raise NotImplementedError


