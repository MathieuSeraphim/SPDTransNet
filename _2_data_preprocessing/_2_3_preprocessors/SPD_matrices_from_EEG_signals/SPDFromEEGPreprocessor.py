import pickle
from multiprocessing import Process

import numpy as np
from _1_configs._1_z_miscellaneous.channel_wise_transformations.utils import get_channel_wise_transformations
from _2_data_preprocessing._2_3_preprocessors.BasePreprocessor import BasePreprocessor
from typing import List, Union
from os import mkdir
from os.path import join, isdir, basename
from _2_data_preprocessing._2_3_preprocessors.ProcessorWrapper import get_preprocessor_from_config
from _2_data_preprocessing._2_3_preprocessors.SPD_matrices_from_EEG_signals import utils
from _2_data_preprocessing._2_3_preprocessors.SPD_matrices_from_EEG_signals.utils import \
    single_window_signal_to_covariance_matrices


class SPDFromEEGPreprocessor(BasePreprocessor):

    def __init__(self, dataset_name: str, **kwargs):
        self.number_of_signals = None
        self.list_of_signals = None
        self.number_of_classes = None
        self.list_of_class_labels = None
        self.epoch_length_in_seconds = None
        self.number_of_subdivisions_per_epoch = None
        self.list_of_signal_preprocessing_strategies = None
        self.number_of_channels = None
        self.list_of_channel_wise_transformations = None
        self.list_of_covariance_estimators = None
        self.list_of_signal_statistics_to_compute = None
        self.compute_signal_statistics_flag = None
        self.compute_recording_wise_covariance_matrices_flag = None
        self.compute_recording_mean_covariance_matrices_flag = None
        self.compute_recording_mean_variance_matrices_flag = None
        self.compute_recording_simple_covariance_matrices_flag = None
        self.compute_recording_simple_variance_matrices_flag = None
        self.compute_recording_mean_signal_statistic_vectors_flag = None
        self.include_epoch_eeg_signals_flag = None
        self.include_recording_eeg_signals_flag = None
        self.covariance_estimator_extra_arguments_as_dict = None
        self.multiprocessing_flag = None
        self.prior_config_preprocessor = None
        self.__initialization_arguments_parsed = False
        super(SPDFromEEGPreprocessor, self).__init__(dataset_name, **kwargs)

    def parse_initialization_arguments(self, eeg_signals: List[str], labels: List[str],
                                       epoch_length_in_seconds: int, number_of_subdivisions_per_epoch: int,
                                       signal_preprocessing_strategies: List[str],
                                       channel_wise_transformations_config_file: str,
                                       channel_wise_transformations_list: List[str],
                                       covariance_estimators: List[str], signal_statistics_as_vectors: List[str],
                                       include_epoch_eeg_signals: bool, compute_recording_mean_matrices: bool,
                                       compute_recording_covariance_matrices: bool, include_recording_eeg_signals: bool,
                                       compute_recording_matrices_no_covariance: bool = False,
                                       random_seed: int = 42, multiprocessing: bool = False,
                                       prior_config_file: Union[str, None] = None):
        assert not self.__initialization_arguments_parsed

        self.number_of_signals = len(eeg_signals)
        assert self.number_of_signals >= 2
        self.list_of_signals = eeg_signals
        self.number_of_classes = len(labels)
        assert self.number_of_classes >= 2
        self.list_of_class_labels = labels
        self.epoch_length_in_seconds = epoch_length_in_seconds
        assert self.epoch_length_in_seconds > 0
        self.number_of_subdivisions_per_epoch = number_of_subdivisions_per_epoch
        assert self.number_of_subdivisions_per_epoch > 0
        self.list_of_signal_preprocessing_strategies = signal_preprocessing_strategies
        channel_wise_transformations = get_channel_wise_transformations(channel_wise_transformations_config_file,
                                                                        channel_wise_transformations_list)
        self.number_of_channels = len(channel_wise_transformations)
        assert self.number_of_channels > 0
        self.list_of_channel_wise_transformations = channel_wise_transformations
        self.list_of_covariance_estimators = covariance_estimators
        assert len(self.list_of_covariance_estimators) > 0
        self.list_of_signal_statistics_to_compute = signal_statistics_as_vectors

        self.compute_signal_statistics_flag = len(self.list_of_signal_statistics_to_compute) > 0

        self.compute_recording_mean_covariance_matrices_flag = compute_recording_mean_matrices
        self.compute_recording_mean_variance_matrices_flag \
            = self.compute_recording_mean_covariance_matrices_flag and compute_recording_matrices_no_covariance

        self.compute_recording_simple_covariance_matrices_flag = compute_recording_covariance_matrices
        self.compute_recording_simple_variance_matrices_flag \
            = self.compute_recording_simple_covariance_matrices_flag and compute_recording_matrices_no_covariance

        self.compute_recording_wise_covariance_matrices_flag = self.compute_recording_mean_covariance_matrices_flag or\
                                                               self.compute_recording_simple_covariance_matrices_flag

        self.compute_recording_mean_signal_statistic_vectors_flag\
            = self.compute_signal_statistics_flag and self.compute_recording_wise_covariance_matrices_flag
        self.include_epoch_eeg_signals_flag = include_epoch_eeg_signals
        self.include_recording_eeg_signals_flag = include_recording_eeg_signals

        self.covariance_estimator_extra_arguments_as_dict = {"random_state": random_seed}

        self.multiprocessing_flag = multiprocessing

        if prior_config_file is not None:
            self.prior_config_preprocessor = get_preprocessor_from_config(prior_config_file)
            assert isinstance(self.prior_config_preprocessor, SPDFromEEGPreprocessor)

        self.__initialization_arguments_parsed = True

    # Progressively deeper dicts for data storage
    # Use _z_miscellaneous.standalone_tests.nested_dicts_and_lists_exploration.recursive_exploration_from_pickle_file
    # on the output to see the structure in details
    def preprocess(self, recording_ids_list: Union[List[int], None] = None):
        assert self.__initialization_arguments_parsed
        super(SPDFromEEGPreprocessor, self).preprocess()

        if recording_ids_list is None:
            recording_ids_list = list(range(len(self.list_of_recording_pickle_filepaths)))

        assert len(recording_ids_list) >= 1
        assert len(recording_ids_list) == len(set(recording_ids_list))
        assert 0 <= min(recording_ids_list) <= max(recording_ids_list) < len(self.list_of_recording_pickle_filepaths)

        processes_list = []
        for recording_id in recording_ids_list:
            if self.multiprocessing_flag:
                process_kwargs = {"recording_id": recording_id, "verbose": False}
                process = Process(target=self.per_recording_preprocessing, kwargs=process_kwargs)
                processes_list.append(process)
                process.start()
            else:
                self.per_recording_preprocessing(recording_id, verbose=True)

        for process in processes_list:
            process.join()

    def per_recording_preprocessing(self, recording_id: int, verbose: bool = True):
        recording_pickle_filepath = self.list_of_recording_pickle_filepaths[recording_id]
        recording_name = basename(recording_pickle_filepath)[:-4]

        recording_folder = join(self.preprocessed_data_folder, recording_name)
        if not isdir(recording_folder):
            mkdir(recording_folder)

        epoch_files_folder = join(recording_folder, "epochs")
        if not isdir(epoch_files_folder):
            mkdir(epoch_files_folder)

        print("\nStarted preprocessing recording %s...\n" % recording_name)

        with open(recording_pickle_filepath, "rb") as f:
            recording_data = pickle.load(f)

        recording_data_keys = list(recording_data.keys())
        recording_sampling_frequency = recording_data["Fs"]
        recording_groundtruth = recording_data["hypno"]
        for epoch_groundtruth in recording_groundtruth:
            assert 0 <= epoch_groundtruth < self.number_of_classes
        recording_available_signals = recording_data["EEG Signals"]

        signal_as_list = []
        for electrode in self.list_of_signals:
            assert electrode in recording_available_signals and electrode in recording_data_keys
            signal_as_list.append(recording_data[electrode])

        # shape (number_of_signals, recording_signal_length_in_steps)
        signal = np.stack(signal_as_list, axis=0)

        assert len(signal.shape) == 2
        assert signal.shape[0] == self.number_of_signals
        recording_size_in_steps = signal.shape[1]
        epoch_size_in_steps = recording_sampling_frequency * self.epoch_length_in_seconds
        epoch_subdivision_size_in_steps = epoch_size_in_steps / self.number_of_subdivisions_per_epoch

        assert epoch_size_in_steps == int(epoch_size_in_steps)
        assert epoch_subdivision_size_in_steps == int(epoch_subdivision_size_in_steps)
        epoch_size_in_steps = int(epoch_size_in_steps)
        epoch_subdivision_size_in_steps = int(epoch_subdivision_size_in_steps)

        number_of_epochs_in_recording = len(recording_groundtruth)
        assert number_of_epochs_in_recording == recording_size_in_steps / epoch_size_in_steps

        recording_computed_data = {}
        for signal_preprocessing_strategy in self.list_of_signal_preprocessing_strategies:

            strategy_in_prior_config_flag = False
            if self.prior_config_preprocessor is not None:
                strategy_in_prior_config_flag = (
                            signal_preprocessing_strategy in self.prior_config_preprocessor.list_of_signal_preprocessing_strategies)

            if verbose:
                print("Preprocessing signal with strategy: %s..." % signal_preprocessing_strategy)

            # shape (number_of_signals, recording_signal_length_in_steps)
            preprocessed_signal = utils.signal_preprocessing(signal, signal_preprocessing_strategy)

            recording_computed_data[signal_preprocessing_strategy] = {}

            for channel_id in range(self.number_of_channels):

                channel_transformation_type, channel_transformation_config \
                    = self.list_of_channel_wise_transformations[channel_id]

                channel_in_prior_config_flag = False
                if self.prior_config_preprocessor is not None and strategy_in_prior_config_flag:

                    for prior_channel_id in range(self.prior_config_preprocessor.number_of_channels):
                        prior_channel_transformation_type, prior_channel_transformation_config = \
                            self.prior_config_preprocessor.list_of_channel_wise_transformations[prior_channel_id]

                        if (prior_channel_transformation_type, prior_channel_transformation_config) \
                                == (channel_transformation_type, channel_transformation_config):
                            channel_in_prior_config_flag = True
                            break

                if verbose:
                    print("Further signal preprocessing for channel %d: %s..." % (channel_id, channel_transformation_type))

                # shape (number_of_signals, recording_signal_length_in_steps)
                channel_signal = utils.channel_wise_signal_transformation(
                    preprocessed_signal, channel_transformation_type, channel_transformation_config,
                    sampling_frequency=recording_sampling_frequency)

                # shape (number_of_signals, number_of_epochs_in_recording, number_of_subdivisions_per_epoch, epoch_subdivision_size_in_steps)
                windowed_channel_signal = utils.subdivide_signal(channel_signal, epoch_subdivision_size_in_steps,
                                                                 self.number_of_subdivisions_per_epoch)

                # shape (number_of_epochs_in_recording * number_of_subdivisions_per_epoch, number_of_signals, epoch_subdivision_size_in_steps)
                windowed_channel_signal, number_of_windows_tuple = utils.batch_windowed_signal_reformatting(
                    windowed_channel_signal)
                assert windowed_channel_signal.shape == \
                       (number_of_epochs_in_recording * self.number_of_subdivisions_per_epoch,
                        self.number_of_signals, epoch_subdivision_size_in_steps)
                assert number_of_windows_tuple == (number_of_epochs_in_recording,
                                                   self.number_of_subdivisions_per_epoch)

                recording_computed_data[signal_preprocessing_strategy][channel_id] = {}
                recording_computed_data[signal_preprocessing_strategy][channel_id]["transformation type"] = \
                    channel_transformation_type
                recording_computed_data[signal_preprocessing_strategy][channel_id]["matrices"] = {}
                recording_computed_data[signal_preprocessing_strategy][channel_id]["statistic vectors"] = {}

                prior_epoch_eeg_signals_flag = False
                prior_recording_eeg_signals_flag = False
                if self.prior_config_preprocessor is not None and channel_in_prior_config_flag:
                    prior_epoch_eeg_signals_flag = self.prior_config_preprocessor.include_epoch_eeg_signals_flag
                    prior_recording_eeg_signals_flag = self.prior_config_preprocessor.include_recording_eeg_signals_flag

                if self.include_epoch_eeg_signals_flag and not prior_epoch_eeg_signals_flag:
                    recording_computed_data[signal_preprocessing_strategy][channel_id]["windowed EEG signals"] = \
                        windowed_channel_signal.reshape(number_of_epochs_in_recording,
                                                        self.number_of_subdivisions_per_epoch,
                                                        self.number_of_signals, epoch_subdivision_size_in_steps)

                if (self.include_recording_eeg_signals_flag and not prior_recording_eeg_signals_flag) or self.compute_recording_simple_covariance_matrices_flag:
                    recording_computed_data[signal_preprocessing_strategy][channel_id]["recording EEG signals"] = \
                        channel_signal

                for covariance_estimator in self.list_of_covariance_estimators:

                    prior_covariance_estimation_computations_flag = False
                    if self.prior_config_preprocessor is not None and channel_in_prior_config_flag:
                        prior_covariance_estimation_computations_flag = \
                            (covariance_estimator in self.prior_config_preprocessor.list_of_covariance_estimators)

                    if prior_covariance_estimation_computations_flag:
                        if verbose:
                            print("Covariances using estimator: %s computed in previous preprocessing run. Skipping..." % covariance_estimator)
                        continue

                    if verbose:
                        print("Computing covariances using estimator: %s..." % covariance_estimator)

                    # shape (number_of_epochs_in_recording, number_of_subdivisions_per_epoch, number_of_signals, number_of_signals)
                    covariance_matrices = utils.batch_windowed_signal_to_covariance_matrices(
                        windowed_channel_signal, number_of_windows_tuple, covariance_estimator,
                        self.covariance_estimator_extra_arguments_as_dict)

                    recording_computed_data[signal_preprocessing_strategy][channel_id]["matrices"][
                        covariance_estimator] = covariance_matrices

                for signal_statistic_vector_type in self.list_of_signal_statistics_to_compute:

                    prior_statistic_vector_computations_flag = False
                    if self.prior_config_preprocessor is not None and channel_in_prior_config_flag:
                        prior_statistic_vector_computations_flag = \
                            (
                                        signal_statistic_vector_type in self.prior_config_preprocessor.list_of_signal_statistics_to_compute)

                    if prior_statistic_vector_computations_flag:
                        if verbose:
                            print("Statistic vectors for stat %s computed in previous preprocessing run. Skipping..." % signal_statistic_vector_type)
                        continue

                    if verbose:
                        print("Computing statistic vectors on epoch subwindows for stat %s..." % signal_statistic_vector_type)

                    statistic_vectors = utils.batch_windowed_signal_to_statistic_vectors(
                        windowed_channel_signal, number_of_windows_tuple, signal_statistic_vector_type,
                        sampling_frequency=recording_sampling_frequency)

                    recording_computed_data[signal_preprocessing_strategy][channel_id]["statistic vectors"][
                        signal_statistic_vector_type] = statistic_vectors

        if verbose:
            print("Saving epoch-level preprocessing outputs...")

        for epoch_id in range(number_of_epochs_in_recording):

            epoch_label_id = recording_groundtruth[epoch_id]
            epoch_label = self.list_of_class_labels[epoch_label_id]

            epoch_filename = join(epoch_files_folder, str(epoch_id).zfill(4) + "_" + epoch_label + ".pkl")

            epoch_data_dict = {"epoch recording id": recording_id,
                               "label": epoch_label,
                               "label id": epoch_label_id,
                               "sampling frequency": recording_sampling_frequency,
                               "matrices size": self.number_of_signals,
                               "signals list": self.list_of_signals,
                               "signal preprocessing strategies": []}

            prior_epoch_data_dict = None
            if self.prior_config_preprocessor is not None:
                with open(epoch_filename, "rb") as f:
                    prior_epoch_data_dict = pickle.load(f)
                assert prior_epoch_data_dict["epoch recording id"] == epoch_data_dict["epoch recording id"]
                assert prior_epoch_data_dict["label"] == epoch_data_dict["label"]
                assert prior_epoch_data_dict["label id"] == epoch_data_dict["label id"]
                assert prior_epoch_data_dict["sampling frequency"] == epoch_data_dict["sampling frequency"]
                assert prior_epoch_data_dict["matrices size"] == epoch_data_dict["matrices size"]
                assert prior_epoch_data_dict["signals list"] == epoch_data_dict["signals list"]

            for signal_preprocessing_strategy in self.list_of_signal_preprocessing_strategies:

                signal_preprocessing_strategy_dict = {"strategy": signal_preprocessing_strategy,
                                                      "channels": []}

                prior_signal_preprocessing_strategy_dict = None
                if prior_epoch_data_dict is not None:
                    if signal_preprocessing_strategy in self.prior_config_preprocessor.list_of_signal_preprocessing_strategies:
                        for prior_strategy_dict in prior_epoch_data_dict["signal preprocessing strategies"]:
                            if prior_strategy_dict["strategy"] == signal_preprocessing_strategy:
                                prior_signal_preprocessing_strategy_dict = prior_strategy_dict
                                break
                        assert prior_signal_preprocessing_strategy_dict is not None

                for channel_id in range(self.number_of_channels):

                    channel_dict = {"channel id": channel_id,
                                    "transformation type":
                                        recording_computed_data[signal_preprocessing_strategy][channel_id][
                                            "transformation type"],
                                    "matrices": []}

                    prior_channel_dict = None
                    if prior_signal_preprocessing_strategy_dict is not None:
                        channel_transformation_type, channel_transformation_config \
                            = self.list_of_channel_wise_transformations[channel_id]

                        for prior_channel_id in range(self.prior_config_preprocessor.number_of_channels):
                            prior_channel_transformation_type, prior_channel_transformation_config = \
                                self.prior_config_preprocessor.list_of_channel_wise_transformations[prior_channel_id]

                            if (prior_channel_transformation_type, prior_channel_transformation_config) \
                                    == (channel_transformation_type, channel_transformation_config):
                                prior_channel_dict = prior_signal_preprocessing_strategy_dict["channels"][
                                    prior_channel_id]
                                break

                    for covariance_estimator in self.list_of_covariance_estimators:
                        matrices_dict = None

                        if prior_channel_dict is not None:
                            if covariance_estimator in self.prior_config_preprocessor.list_of_covariance_estimators:
                                for prior_matrices_dict in prior_channel_dict["matrices"]:
                                    if prior_matrices_dict["covariance estimator label"] == covariance_estimator:
                                        matrices_dict = prior_matrices_dict
                                        break
                                assert matrices_dict is not None

                        if matrices_dict is None:
                            matrices_dict = {"covariance estimator label": covariance_estimator,
                                             "data": recording_computed_data[signal_preprocessing_strategy][channel_id]["matrices"][covariance_estimator][epoch_id, :, :, :]}

                        channel_dict["matrices"].append(matrices_dict)

                    if self.compute_signal_statistics_flag:
                        channel_dict["statistic vectors"] = []

                        prior_statistic_vectors_list = None
                        if prior_channel_dict is not None:
                            if self.prior_config_preprocessor.compute_signal_statistics_flag:
                                prior_statistic_vectors_list = prior_channel_dict["statistic vectors"]

                        for signal_statistic_vector_type in self.list_of_signal_statistics_to_compute:
                            statistic_vectors_dict = None

                            if prior_channel_dict is not None:
                                if signal_statistic_vector_type in self.prior_config_preprocessor.list_of_signal_statistics_to_compute:
                                    for prior_statistic_vectors_dict in prior_statistic_vectors_list:
                                        if prior_statistic_vectors_dict["computed statistic"] == signal_statistic_vector_type:
                                            statistic_vectors_dict = prior_statistic_vectors_dict
                                            break
                                    assert statistic_vectors_dict is not None

                            if statistic_vectors_dict is None:
                                statistic_vectors_dict = {"computed statistic": signal_statistic_vector_type,
                                                          "data":
                                                              recording_computed_data[signal_preprocessing_strategy][
                                                                  channel_id]["statistic vectors"][
                                                                  signal_statistic_vector_type][epoch_id, :, :]}
                            channel_dict["statistic vectors"].append(statistic_vectors_dict)

                    if self.include_epoch_eeg_signals_flag:
                        epoch_eeg_signals = None

                        if prior_channel_dict is not None:
                            if self.prior_config_preprocessor.include_epoch_eeg_signals_flag:
                                epoch_eeg_signals = prior_channel_dict["epoch EEG signals"]

                        if epoch_eeg_signals is None:
                            epoch_eeg_signals = recording_computed_data[signal_preprocessing_strategy][channel_id][
                                                    "windowed EEG signals"][epoch_id, :, :, :]

                        channel_dict["epoch EEG signals"] = epoch_eeg_signals

                    signal_preprocessing_strategy_dict["channels"].append(channel_dict)

                epoch_data_dict["signal preprocessing strategies"].append(signal_preprocessing_strategy_dict)

            with open(epoch_filename, 'wb') as f:
                pickle.dump(epoch_data_dict, f)
            del epoch_data_dict, prior_epoch_data_dict

        if self.compute_recording_wise_covariance_matrices_flag:

            if verbose:
                print("Computing and saving recording-level data...")

            recording_filename = join(recording_folder, "recording_means.pkl")

            recording_data_dict = {"sampling frequency": recording_sampling_frequency,
                                   "matrix size": self.number_of_signals,
                                   "signals list": self.list_of_signals,
                                   "signal preprocessing strategies": []}

            prior_recording_data_dict = None
            if self.prior_config_preprocessor is not None:
                if self.prior_config_preprocessor.compute_recording_wise_covariance_matrices_flag:
                    with open(recording_filename, "rb") as f:
                        prior_recording_data_dict = pickle.load(f)
                    assert prior_recording_data_dict["sampling frequency"] == recording_data_dict["sampling frequency"]
                    assert prior_recording_data_dict["matrix size"] == recording_data_dict["matrix size"]
                    assert prior_recording_data_dict["signals list"] == recording_data_dict["signals list"]

            for signal_preprocessing_strategy in self.list_of_signal_preprocessing_strategies:

                signal_preprocessing_strategy_dict = {"strategy": signal_preprocessing_strategy,
                                                      "channels": []}

                prior_signal_preprocessing_strategy_dict = None
                if prior_recording_data_dict is not None:
                    if signal_preprocessing_strategy in self.prior_config_preprocessor.list_of_signal_preprocessing_strategies:
                        for prior_strategy_dict in prior_recording_data_dict["signal preprocessing strategies"]:
                            if prior_strategy_dict["strategy"] == signal_preprocessing_strategy:
                                prior_signal_preprocessing_strategy_dict = prior_strategy_dict
                                break
                        assert prior_signal_preprocessing_strategy_dict is not None

                for channel_id in range(self.number_of_channels):
                    channel_dict = {"channel id": channel_id,
                                    "transformation type":
                                        recording_computed_data[signal_preprocessing_strategy][channel_id][
                                            "transformation type"],
                                    "matrices": []}

                    prior_channel_dict = None
                    if prior_signal_preprocessing_strategy_dict is not None:
                        channel_transformation_type, channel_transformation_config \
                            = self.list_of_channel_wise_transformations[channel_id]

                        for prior_channel_id in range(self.prior_config_preprocessor.number_of_channels):
                            prior_channel_transformation_type, prior_channel_transformation_config = \
                                self.prior_config_preprocessor.list_of_channel_wise_transformations[prior_channel_id]

                            if (prior_channel_transformation_type, prior_channel_transformation_config) \
                                    == (channel_transformation_type, channel_transformation_config):
                                prior_channel_dict = prior_signal_preprocessing_strategy_dict["channels"][
                                    prior_channel_id]
                                break

                    if verbose:
                        print("For strategy %s, channel %d:" % (signal_preprocessing_strategy, channel_id))

                    for covariance_estimator in self.list_of_covariance_estimators:
                        matrices_dict = {"covariance estimator label": covariance_estimator}

                        if self.compute_recording_mean_covariance_matrices_flag:
                            mean_matrix = None
                            mean_variance_matrix = None

                            if verbose:
                                print("  Processing affine invariant mean matrix for estimator %s..." % covariance_estimator)

                            if prior_channel_dict is not None and self.prior_config_preprocessor.compute_recording_mean_covariance_matrices_flag:
                                if covariance_estimator in self.prior_config_preprocessor.list_of_covariance_estimators:
                                    for prior_matrices_dict in prior_channel_dict["matrices"]:
                                        if prior_matrices_dict["covariance estimator label"] == covariance_estimator:
                                            mean_matrix = prior_matrices_dict["mean matrix"]
                                            if self.prior_config_preprocessor.compute_recording_mean_variance_matrices_flag:
                                                mean_variance_matrix = prior_matrices_dict["mean variance matrix"]
                                            break
                                    assert mean_matrix is not None

                            if mean_matrix is None:
                                mean_matrix = utils.batch_spd_matrices_affine_invariant_mean(
                                    recording_computed_data[signal_preprocessing_strategy][channel_id]["matrices"][
                                        covariance_estimator])

                            if self.compute_recording_mean_variance_matrices_flag and mean_variance_matrix is None:
                                mean_variance_matrix = utils.batch_spd_matrices_affine_invariant_mean(
                                    recording_computed_data[signal_preprocessing_strategy][channel_id]["matrices"][
                                        covariance_estimator], True)

                            matrices_dict["mean matrix"] = mean_matrix
                            if self.compute_recording_mean_variance_matrices_flag:
                                matrices_dict["mean variance matrix"] = mean_variance_matrix

                        if self.compute_recording_simple_covariance_matrices_flag:
                            recording_covariance_matrix = None
                            recording_variance_matrix = None

                            if verbose:
                                print("  Processing recording-wise covariance matrix for estimator %s..." % covariance_estimator)

                            if prior_channel_dict is not None and self.prior_config_preprocessor.compute_recording_simple_covariance_matrices_flag:
                                if covariance_estimator in self.prior_config_preprocessor.list_of_covariance_estimators:
                                    for prior_matrices_dict in prior_channel_dict["matrices"]:
                                        if prior_matrices_dict["covariance estimator label"] == covariance_estimator:
                                            recording_covariance_matrix = prior_matrices_dict["recording covariance matrix"]
                                            if self.prior_config_preprocessor.compute_recording_simple_variance_matrices_flag:
                                                recording_variance_matrix = prior_matrices_dict["recording variance matrix"]
                                            break
                                    assert recording_covariance_matrix is not None

                            if recording_covariance_matrix is None:
                                recording_covariance_matrix = single_window_signal_to_covariance_matrices(
                                    recording_computed_data[signal_preprocessing_strategy][channel_id][
                                        "recording EEG signals"],
                                    covariance_estimator, self.covariance_estimator_extra_arguments_as_dict)

                            if self.compute_recording_simple_variance_matrices_flag and recording_variance_matrix is None:
                                assert len(recording_covariance_matrix.shape) == 2
                                recording_variance_matrix = np.diag(np.diag(recording_covariance_matrix))  # Somehow, this is a legitimate way to set non-diagonal elements to 0

                            matrices_dict["recording covariance matrix"] = recording_covariance_matrix
                            if self.compute_recording_mean_variance_matrices_flag:
                                matrices_dict["recording variance matrix"] = recording_variance_matrix

                        channel_dict["matrices"].append(matrices_dict)

                    if self.compute_recording_mean_signal_statistic_vectors_flag:
                        channel_dict["statistic vectors"] = []

                        prior_statistic_vectors_list = None
                        if prior_channel_dict is not None:
                            if self.prior_config_preprocessor.compute_signal_statistics_flag:
                                prior_statistic_vectors_list = prior_channel_dict["statistic vectors"]

                        if verbose:
                            print("  Processing mean statistic vectors...")

                        for signal_statistic_vector_type in self.list_of_signal_statistics_to_compute:
                            statistic_vectors_dict = None

                            if prior_channel_dict is not None:
                                if signal_statistic_vector_type in self.prior_config_preprocessor.list_of_signal_statistics_to_compute:
                                    for prior_statistic_vectors_dict in prior_statistic_vectors_list:
                                        if prior_statistic_vectors_dict["computed statistic"] == signal_statistic_vector_type:
                                            statistic_vectors_dict = prior_statistic_vectors_dict
                                            break
                                    assert statistic_vectors_dict is not None

                            if statistic_vectors_dict is None:
                                statistic_vectors_dict = {"computed statistic": signal_statistic_vector_type}
                                mean_vector = utils.batch_vectors_euclidean_mean(
                                    recording_computed_data[signal_preprocessing_strategy][channel_id][
                                        "statistic vectors"][signal_statistic_vector_type])
                                statistic_vectors_dict["mean vector"] = mean_vector

                            channel_dict["statistic vectors"].append(statistic_vectors_dict)

                    if self.include_recording_eeg_signals_flag:
                        recording_eeg_signals = None

                        if prior_channel_dict is not None:
                            if self.prior_config_preprocessor.include_recording_eeg_signals_flag:
                                recording_eeg_signals = prior_channel_dict["recording EEG signals"]

                        if recording_eeg_signals is None:
                            recording_eeg_signals = recording_computed_data[signal_preprocessing_strategy][channel_id][
                                "recording EEG signals"]

                        channel_dict["recording EEG signals"] = recording_eeg_signals

                    signal_preprocessing_strategy_dict["channels"].append(channel_dict)

                recording_data_dict["signal preprocessing strategies"].append(signal_preprocessing_strategy_dict)

            with open(recording_filename, 'wb') as f:
                pickle.dump(recording_data_dict, f)
            del recording_data_dict, prior_recording_data_dict

        print("\nEnded preprocessing of recording %s.\n" % recording_name)


def main(config_file: str, recording_ids_list: Union[List[int], None] = None):
    preprocessor = get_preprocessor_from_config(config_file)
    preprocessor.preprocess(recording_ids_list)


if __name__ == "__main__":
    main("SPD_matrices_from_EEG_MASS_dataset_ICASSP_signals_config.yaml")


