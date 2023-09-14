import pickle
from typing import List, Tuple, Any
import numpy as np
import torch
from _2_data_preprocessing._2_3_preprocessors.SPD_matrices_from_EEG_signals.utils import \
    batch_remove_non_diagonal_elements_from_matrices
from _3_data_management._3_1_data_readers.BaseDataReader import BaseDataReader
from _2_data_preprocessing._2_3_preprocessors.SPD_matrices_from_EEG_signals.SPDFromEEGPreprocessor import SPDFromEEGPreprocessor
from os.path import join, isfile, basename, dirname, isdir
from os import listdir


class SPDFromEEGDataReader(BaseDataReader):

    COMPATIBLE_PREPROCESSOR_CLASS = SPDFromEEGPreprocessor
    
    def __init__(self, preprocessing_config_file: str, **kwargs):
        super(SPDFromEEGDataReader, self).__init__(preprocessing_config_file, **kwargs)

    def parse_preprocessor_attributes(self):

        self.dataset_name = self.corresponding_preprocessor_object.dataset_name
        self.transformation_config_name = self.corresponding_preprocessor_object.transformation_config_name

        self.list_of_signals = self.corresponding_preprocessor_object.list_of_signals
        self.number_of_signals = len(self.list_of_signals)
        self.list_of_class_labels = self.corresponding_preprocessor_object.list_of_class_labels
        self.list_of_signal_preprocessing_strategies = self.corresponding_preprocessor_object.list_of_signal_preprocessing_strategies
        self.list_of_channel_wise_transformations = self.corresponding_preprocessor_object.list_of_channel_wise_transformations
        self.list_of_covariance_estimators = self.corresponding_preprocessor_object.list_of_covariance_estimators
        self.list_of_signal_statistics = self.corresponding_preprocessor_object.list_of_signal_statistics_to_compute

        self.include_signal_statistics_flag = self.corresponding_preprocessor_object.compute_signal_statistics_flag
        self.include_epoch_eeg_signals_flag = self.corresponding_preprocessor_object.include_epoch_eeg_signals_flag

        self.include_recording_wise_covariance_matrices_flag = self.corresponding_preprocessor_object.compute_recording_wise_covariance_matrices_flag
        self.include_recording_mean_covariance_matrices_flag = self.corresponding_preprocessor_object.compute_recording_mean_covariance_matrices_flag
        self.include_recording_mean_variance_matrices_flag = self.corresponding_preprocessor_object.compute_recording_mean_variance_matrices_flag
        self.include_recording_simple_covariance_matrices_flag = self.corresponding_preprocessor_object.compute_recording_simple_covariance_matrices_flag
        self.include_recording_simple_variance_matrices_flag = self.corresponding_preprocessor_object.compute_recording_simple_variance_matrices_flag
        self.include_recording_mean_signal_statistic_vectors = self.corresponding_preprocessor_object.compute_recording_mean_signal_statistic_vectors_flag
        self.include_recording_eeg_signals_flag = self.corresponding_preprocessor_object.include_recording_eeg_signals_flag

        self.preprocessed_data_folder = self.corresponding_preprocessor_object.preprocessed_data_folder
        assert isdir(self.preprocessed_data_folder)

        recording_names = listdir(self.preprocessed_data_folder)
        recording_names.sort()
        number_of_recordings = len(recording_names)
        assert number_of_recordings > 0

        self.all_epochs_metadata_list = []
        self.all_epochs_labels_list = []
        self.all_recordings_metadata_list = []

        number_of_epochs_so_far = 0

        for recording_id in range(number_of_recordings):
            recording_dict = {}

            recording_name = recording_names[recording_id]
            recording_folder = join(self.preprocessed_data_folder, recording_name)
            recording_dict["recording name"] = recording_name

            if self.include_recording_wise_covariance_matrices_flag:
                recording_data_file = join(recording_folder, "recording_means.pkl")
                assert isfile(recording_data_file)
                recording_dict["data file"] = recording_data_file

            epochs_folder = join(recording_folder, "epochs")
            epochs_base_filenames = listdir(epochs_folder)
            epochs_base_filenames.sort()
            number_of_epochs_in_recording = len(epochs_base_filenames)
            assert number_of_epochs_in_recording > 0

            predicted_last_epoch_global_id_plus_one = number_of_epochs_so_far + number_of_epochs_in_recording
            recording_dict["epoch ids range"] = (number_of_epochs_so_far, predicted_last_epoch_global_id_plus_one)

            for epoch_id_in_recording in range(number_of_epochs_in_recording):
                epoch_base_filename = epochs_base_filenames[epoch_id_in_recording]
                assert epoch_base_filename[-4:] == ".pkl"
                epoch_dict = {}

                epoch_filename = join(epochs_folder, epoch_base_filename)
                assert isfile(epoch_filename)
                epoch_dict["data file"] = epoch_filename

                epoch_name = epoch_base_filename[:-4]
                epoch_id_from_file, epoch_label = epoch_name.split("_")
                assert int(epoch_id_from_file) == epoch_id_in_recording
                assert epoch_label in self.list_of_class_labels
                epoch_dict["epoch label from file"] = epoch_label
                epoch_label_id = self.list_of_class_labels.index(epoch_label)

                self.all_epochs_metadata_list.append(epoch_dict)
                self.all_epochs_labels_list.append(epoch_label_id)
                number_of_epochs_so_far += 1

            assert number_of_epochs_so_far == predicted_last_epoch_global_id_plus_one\
                   == len(self.all_epochs_metadata_list) == len(self.all_epochs_labels_list)
            self.all_recordings_metadata_list.append(recording_dict)

        del self.corresponding_preprocessor_object

    # No specific initialization arguments for this data reader
    def parse_initialization_arguments(self, **kwargs):
        pass

    def setup(self, eeg_signals: List[str], labels: List[str], signal_preprocessing_strategy: str,
              channel_wise_transformations: List[Tuple[str, Any]], covariance_estimator: str,
              statistic_vectors_to_return: List[str], return_epoch_eeg_signals: bool,
              return_recording_wise_matrices: str, return_recording_eeg_signals: bool, no_covariances: bool = False):
        assert self.list_of_signals == eeg_signals
        assert self.list_of_class_labels == labels

        return_recording_mean_matrices = False
        return_recording_covariance_matrices = False
        assert return_recording_wise_matrices in ["none", "affine_invariant_mean", "simple_covariance"]
        if return_recording_wise_matrices == "affine_invariant_mean":
            return_recording_mean_matrices = True
        elif return_recording_wise_matrices == "simple_covariance":
            return_recording_covariance_matrices = True

        self.return_variance_matrices = no_covariances

        assert signal_preprocessing_strategy in self.list_of_signal_preprocessing_strategies
        self.output_signal_preprocessing_strategy = signal_preprocessing_strategy

        self.output_list_of_channel_transformation_indices = []
        for transformation in channel_wise_transformations:

            # Because a list and tuple with the same contents aren't equal in Python
            try:
                assert transformation in self.list_of_channel_wise_transformations
            except:
                try:
                    transformation = list(transformation)
                    assert transformation in self.list_of_channel_wise_transformations
                except:
                    transformation = tuple(transformation)
                    assert transformation in self.list_of_channel_wise_transformations

            transformation_channel_id = self.list_of_channel_wise_transformations.index(transformation)
            self.output_list_of_channel_transformation_indices.append(transformation_channel_id)

        assert covariance_estimator in self.list_of_covariance_estimators
        self.output_covariance_estimator = covariance_estimator

        for statistics_vector_to_return in statistic_vectors_to_return:
            assert statistics_vector_to_return in self.list_of_signal_statistics
        self.output_statistic_vectors = False
        if len(statistic_vectors_to_return) > 0:
            self.output_statistic_vectors = True
        self.output_statistic_vectors_list = statistic_vectors_to_return

        self.output_epoch_eeg_signals = False
        if return_epoch_eeg_signals:
            assert self.include_epoch_eeg_signals_flag
            self.output_epoch_eeg_signals = True

        self.setup_output_recording_mean_matrices = False
        self.setup_output_recording_covariance_matrices = False
        self.setup_output_recording_mean_statistic_vectors = False
        if return_recording_mean_matrices or return_recording_covariance_matrices:
            assert self.include_recording_wise_covariance_matrices_flag

            if return_recording_mean_matrices:
                assert self.include_recording_mean_covariance_matrices_flag
                self.setup_output_recording_mean_matrices = True
                if self.return_variance_matrices:
                    assert self.include_recording_mean_variance_matrices_flag

            if return_recording_covariance_matrices:
                assert self.include_recording_simple_covariance_matrices_flag
                self.setup_output_recording_covariance_matrices = True
                if self.return_variance_matrices:
                    assert self.include_recording_simple_variance_matrices_flag

            assert not (self.setup_output_recording_mean_matrices and self.setup_output_recording_covariance_matrices)

            if self.output_statistic_vectors:
                self.setup_output_recording_mean_statistic_vectors = True

        self.setup_output_recording_eeg_signals = False
        if return_recording_eeg_signals:
            assert self.include_recording_eeg_signals_flag
            self.setup_output_recording_eeg_signals = True

        return self.all_epochs_labels_list, self.setup_output_recording_wise_data()

    def setup_output_recording_wise_data(self):
        recording_wise_data_list = []

        for recording_metadata_dict in self.all_recordings_metadata_list:
            recording_data_dict = {"epoch ids range": recording_metadata_dict["epoch ids range"]}

            if self.setup_output_recording_mean_matrices or self.setup_output_recording_covariance_matrices\
                    or self.setup_output_recording_eeg_signals:
                with open(recording_metadata_dict["data file"], "rb") as f:
                    recording_data = pickle.load(f)

                signals_list = recording_data["signals list"]
                matrix_size = recording_data["matrix size"]
                sampling_frequency = recording_data["sampling frequency"]
                assert signals_list == self.list_of_signals
                assert matrix_size == self.number_of_signals

                recording_eeg_signals_per_channel_list = []
                recording_wise_matrices_per_channel_list = []
                recording_mean_vectors_per_statistic_per_channel_list = []

                wanted_signal_preprocessing_strategy_index = -1
                for signal_preprocessing_strategy_index in range(len(recording_data["signal preprocessing strategies"])):
                    signal_preprocessing_strategy_dict = recording_data["signal preprocessing strategies"][signal_preprocessing_strategy_index]
                    if signal_preprocessing_strategy_dict["strategy"] == self.output_signal_preprocessing_strategy:
                        wanted_signal_preprocessing_strategy_index = signal_preprocessing_strategy_index
                        break
                assert wanted_signal_preprocessing_strategy_index >= 0
                wanted_strategy_dict = recording_data["signal preprocessing strategies"][wanted_signal_preprocessing_strategy_index]

                channel_dicts_list = wanted_strategy_dict["channels"]
                for channel_id in self.output_list_of_channel_transformation_indices:
                    channel_dict = channel_dicts_list[channel_id]
                    assert channel_dict["channel id"] == channel_id
                    assert channel_dict["transformation type"] == self.list_of_channel_wise_transformations[channel_id][0]

                    if self.setup_output_recording_eeg_signals:
                        eeg_signals = channel_dict["recording EEG signals"]
                        assert len(eeg_signals.shape) == 2
                        assert eeg_signals.shape[0] == self.number_of_signals
                        recording_eeg_signals_per_channel_list.append(eeg_signals)

                    if self.setup_output_recording_mean_matrices or self.setup_output_recording_covariance_matrices:

                        estimator_dicts_list = channel_dict["matrices"]
                        wanted_estimator_index = -1
                        for estimator_index in range(len(estimator_dicts_list)):
                            estimator_dict = estimator_dicts_list[estimator_index]
                            if estimator_dict["covariance estimator label"] == self.output_covariance_estimator:
                                wanted_estimator_index = estimator_index
                                break
                        assert wanted_estimator_index >= 0
                        wanted_estimator_dict = estimator_dicts_list[wanted_estimator_index]

                        if self.setup_output_recording_mean_matrices:
                            if self.return_variance_matrices:
                                matrix = wanted_estimator_dict["mean variance matrix"]
                            else:
                                matrix = wanted_estimator_dict["mean matrix"]
                        else:
                            assert self.setup_output_recording_covariance_matrices
                            if self.return_variance_matrices:
                                matrix = wanted_estimator_dict["recording variance matrix"]
                            else:
                                matrix = wanted_estimator_dict["recording covariance matrix"]

                        assert matrix.shape == (self.number_of_signals, self.number_of_signals)
                        recording_wise_matrices_per_channel_list.append(matrix)

                    if self.setup_output_recording_mean_statistic_vectors:
                        recording_mean_vectors_per_statistic_dict = {}
                        number_of_statistics = len(self.output_statistic_vectors_list)
                        statistics_counter = 0

                        statistic_vectors_dicts_list = channel_dict["statistic vectors"]
                        for statistic_vectors_dict in statistic_vectors_dicts_list:
                            if statistic_vectors_dict["computed statistic"] in self.output_statistic_vectors_list:
                                mean_vector = statistic_vectors_dict["mean vector"]
                                assert mean_vector.shape == (self.number_of_signals,)
                                recording_mean_vectors_per_statistic_dict[statistic_vectors_dict["computed statistic"]]\
                                    = mean_vector
                                statistics_counter += 1

                        assert statistics_counter == number_of_statistics
                        recording_mean_vectors_per_statistic_per_channel_list.append(recording_mean_vectors_per_statistic_dict)

            if self.setup_output_recording_eeg_signals:
                recording_data_dict["sampling frequency"] = sampling_frequency
                recording_data_dict["EEG signals"] = torch.Tensor(np.stack(recording_eeg_signals_per_channel_list, axis=0))

            if self.setup_output_recording_mean_matrices or self.setup_output_recording_covariance_matrices:
                recording_data_dict["recording-wise matrices"] = torch.Tensor(np.stack(recording_wise_matrices_per_channel_list, axis=0))

            if self.setup_output_recording_mean_statistic_vectors:
                recording_mean_vectors_per_statistic_combined_dict = {}
                for statistic in self.output_statistic_vectors_list:
                    recording_mean_vectors_per_statistic_combined_dict[statistic] = []

                for channel_mean_vectors_per_statistic_dict in recording_mean_vectors_per_statistic_per_channel_list:
                    for statistic in channel_mean_vectors_per_statistic_dict.keys():
                        recording_mean_vectors_per_statistic_combined_dict[statistic].append(channel_mean_vectors_per_statistic_dict[statistic])

                for statistic in self.output_statistic_vectors_list:
                    recording_mean_vectors_per_statistic_combined_dict[statistic] = torch.Tensor(np.stack(recording_mean_vectors_per_statistic_combined_dict[statistic], axis=0))

                recording_data_dict["mean statistic vectors"] = recording_mean_vectors_per_statistic_combined_dict

            recording_wise_data_list.append(recording_data_dict)

        return recording_wise_data_list

    def get_element_data(self, element_id: int):
        epoch_data_dict = {}

        epoch_metadata_dict = self.all_epochs_metadata_list[element_id]
        epoch_groundtruth = self.all_epochs_labels_list[element_id]
        epoch_label_from_file = epoch_metadata_dict["epoch label from file"]
        recording_name_from_filename = basename(dirname(dirname(epoch_metadata_dict["data file"])))

        with open(epoch_metadata_dict["data file"], "rb") as f:
            epoch_data = pickle.load(f)

        recording_id = epoch_data["epoch recording id"]
        label = epoch_data["label"]
        label_id = epoch_data["label id"]
        matrices_size = epoch_data["matrices size"]
        signals_list = epoch_data["signals list"]
        sampling_frequency = epoch_data["sampling frequency"]
        assert self.all_recordings_metadata_list[recording_id]["recording name"] == recording_name_from_filename
        assert signals_list == self.list_of_signals
        assert matrices_size == self.number_of_signals
        assert label in self.list_of_class_labels
        assert label == epoch_label_from_file
        assert self.list_of_class_labels.index(label) == label_id == epoch_groundtruth

        epoch_data_dict["recording id"] = recording_id
        epoch_data_dict["label id"] = label_id
        epoch_data_dict["label"] = label

        epoch_eeg_signals_per_channel_list = []
        epoch_matrices_per_channel_list = []
        epoch_vectors_per_statistic_per_channel_list = []

        wanted_signal_preprocessing_strategy_index = -1
        for signal_preprocessing_strategy_index in range(len(epoch_data["signal preprocessing strategies"])):
            signal_preprocessing_strategy_dict = epoch_data["signal preprocessing strategies"][signal_preprocessing_strategy_index]
            if signal_preprocessing_strategy_dict["strategy"] == self.output_signal_preprocessing_strategy:
                wanted_signal_preprocessing_strategy_index = signal_preprocessing_strategy_index
                break
        assert wanted_signal_preprocessing_strategy_index >= 0
        wanted_strategy_dict = epoch_data["signal preprocessing strategies"][wanted_signal_preprocessing_strategy_index]

        channel_dicts_list = wanted_strategy_dict["channels"]
        for channel_id in self.output_list_of_channel_transformation_indices:
            channel_dict = channel_dicts_list[channel_id]
            assert channel_dict["channel id"] == channel_id
            assert channel_dict["transformation type"] == self.list_of_channel_wise_transformations[channel_id][0]

            estimator_dicts_list = channel_dict["matrices"]
            wanted_estimator_index = -1
            for estimator_index in range(len(estimator_dicts_list)):
                estimator_dict = estimator_dicts_list[estimator_index]
                if estimator_dict["covariance estimator label"] == self.output_covariance_estimator:
                    wanted_estimator_index = estimator_index
                    break
            assert wanted_estimator_index >= 0
            wanted_estimator_dict = estimator_dicts_list[wanted_estimator_index]

            matrices = wanted_estimator_dict["data"]
            if self.return_variance_matrices:
                matrices = batch_remove_non_diagonal_elements_from_matrices(matrices)

            assert len(matrices.shape) == 3
            assert matrices.shape[1] == matrices.shape[2] == self.number_of_signals
            epoch_matrices_per_channel_list.append(matrices)

            if self.output_epoch_eeg_signals:
                eeg_signals = channel_dict["epoch EEG signals"]
                assert len(eeg_signals.shape) == 3
                assert eeg_signals.shape[1] == self.number_of_signals
                epoch_eeg_signals_per_channel_list.append(eeg_signals)
                
            if self.output_statistic_vectors:
                epoch_vectors_per_statistic_dict = {}
                number_of_statistics = len(self.output_statistic_vectors_list)
                statistics_counter = 0

                statistic_vectors_dicts_list = channel_dict["statistic vectors"]
                for statistic_vectors_dict in statistic_vectors_dicts_list:
                    if statistic_vectors_dict["computed statistic"] in self.output_statistic_vectors_list:
                        statistic_vectors = statistic_vectors_dict["data"]
                        assert len(statistic_vectors.shape) == 2
                        assert statistic_vectors.shape[1] == self.number_of_signals
                        epoch_vectors_per_statistic_dict[statistic_vectors_dict["computed statistic"]] \
                            = statistic_vectors
                        statistics_counter += 1

                assert statistics_counter == number_of_statistics
                epoch_vectors_per_statistic_per_channel_list.append(epoch_vectors_per_statistic_dict)
                
        epoch_data_dict["matrices"] = torch.Tensor(np.stack(epoch_matrices_per_channel_list, axis=0))
        
        if self.output_epoch_eeg_signals:
            epoch_data_dict["sampling frequency"] = sampling_frequency
            epoch_data_dict["EEG signals"] = torch.Tensor(np.stack(epoch_eeg_signals_per_channel_list, axis=0))
            
        if self.output_statistic_vectors:
            epoch_vectors_per_statistic_combined_dict = {}
            for statistic in self.output_statistic_vectors_list:
                epoch_vectors_per_statistic_combined_dict[statistic] = []

            for channel_vectors_per_statistic_dict in epoch_vectors_per_statistic_per_channel_list:
                for statistic in channel_vectors_per_statistic_dict.keys():
                    epoch_vectors_per_statistic_combined_dict[statistic].append(channel_vectors_per_statistic_dict[statistic])

            for statistic in self.output_statistic_vectors_list:
                epoch_vectors_per_statistic_combined_dict[statistic] = torch.Tensor(np.stack(epoch_vectors_per_statistic_combined_dict[statistic], axis=0))

            epoch_data_dict["statistic vectors"] = epoch_vectors_per_statistic_combined_dict
            
        return epoch_data_dict
        

            
        














