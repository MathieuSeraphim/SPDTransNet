import numpy as np
import torch
from torch.utils.data import Dataset
from _3_data_management._3_1_data_readers.DataReaderWrapper import get_data_reader_from_config
from _3_data_management._3_1_data_readers.SPD_matrices_from_EEG_signals.SPDFromEEGDataReader import SPDFromEEGDataReader
from typing import List, Tuple, Any, Union
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


class SPDFromEEGDataset(Dataset):

    COMPATIBLE_DATAREADER_CLASS = SPDFromEEGDataReader

    def __init__(self, eeg_signals: List[str], labels: List[str], data_reader_config_file: str):

        self.__setup_done_flag = False

        self.data_reader = get_data_reader_from_config(data_reader_config_file)
        assert isinstance(self.data_reader, self.COMPATIBLE_DATAREADER_CLASS)

        get_recording_eeg_signals = False
        self.data_reader_setup_kwargs = {"eeg_signals": eeg_signals,
                                         "labels": labels,
                                         "return_recording_eeg_signals": get_recording_eeg_signals}

        self.labels_list = labels
        self.number_of_classes = len(labels)
        assert self.number_of_classes > 0
        self.label_ids_list = list(range(self.number_of_classes))
        self.matrices_size = len(eeg_signals)
        self.transfer_statistic_vectors_flag = False
        self.transfer_eeg_epochs_flag = False

        self.number_of_channels = -1
        self.recording_indices = []
        self.statistic_vectors_list = []
        self.transfer_recording_wise_matrices_flag = False
        self.extra_epochs_on_each_side = -1
        self.sequences_of_epochs_length = -1
        self.recording_wise_data_list = []
        self.dataset_epochs_global_indices_list = []
        self.dataset_epochs_labels_list = []
        self.dataset_length = -1

    def setup(self, recording_indices: List[int], extra_epochs_on_each_side: int, signal_preprocessing_strategy: str,
              channel_wise_transformations: List[Tuple[str, Any]], covariance_estimator: str,
              statistic_vectors_for_matrix_augmentation: List[str], transfer_recording_wise_matrices: bool,
              rebalance_set_by_oversampling: bool = False, clip_recordings_by_amount: Union[int, None] = None,
              use_recording_wise_simple_covariances: bool = False, no_covariances: bool = False,
              get_epoch_eeg_signals: bool = False, random_seed: int = 42):
        assert not self.__setup_done_flag

        statistic_vectors_for_matrix_augmentation = [statistic_vector_name
                                                     for statistic_vector_name in statistic_vectors_for_matrix_augmentation
                                                     if statistic_vector_name != ""]

        self.extra_epochs_on_each_side = extra_epochs_on_each_side
        self.number_of_channels = len(channel_wise_transformations)
        self.statistic_vectors_list = statistic_vectors_for_matrix_augmentation
        if len(self.statistic_vectors_list) > 0:
            self.transfer_statistic_vectors_flag = True
        self.transfer_recording_wise_matrices_flag = transfer_recording_wise_matrices
        self.transfer_eeg_epochs_flag = get_epoch_eeg_signals
        self.recording_indices = recording_indices
        assert len(self.recording_indices) > 0
        assert self.extra_epochs_on_each_side >= 0
        self.sequences_of_epochs_length = 1 + 2 * self.extra_epochs_on_each_side

        return_recording_wise_matrices = "none"
        if transfer_recording_wise_matrices:
            if use_recording_wise_simple_covariances:
                return_recording_wise_matrices = "simple_covariance"
            else:
                return_recording_wise_matrices = "affine_invariant_mean"

        all_epochs_labels_list, recording_wise_data_list = self.data_reader.setup(
            **self.data_reader_setup_kwargs,
            signal_preprocessing_strategy=signal_preprocessing_strategy,
            channel_wise_transformations=channel_wise_transformations,
            covariance_estimator=covariance_estimator,
            statistic_vectors_to_return=statistic_vectors_for_matrix_augmentation,
            return_epoch_eeg_signals=get_epoch_eeg_signals,
            return_recording_wise_matrices=return_recording_wise_matrices,
            no_covariances=no_covariances
        )

        # Making sure that if statistic vectors have been loaded, their mean was as well
        if self.transfer_recording_wise_matrices_flag:
            assert self.data_reader.output_statistic_vectors == self.data_reader.setup_output_recording_mean_statistic_vectors == self.transfer_statistic_vectors_flag

        if self.transfer_eeg_epochs_flag:
            assert self.data_reader.include_epoch_eeg_signals_flag

        self.recording_wise_data_list = [recording_wise_data_list[recording_index] for recording_index in self.recording_indices]

        # Sequence-to-epoch classification -> sequences constructed around a central epoch, with the first and last
        # l = extra_epochs_on_each_side not classified
        # This clipping may be extended to an arbitrary length
        amount_of_epochs_to_remove_on_each_side = self.extra_epochs_on_each_side
        if clip_recordings_by_amount is not None:
            assert amount_of_epochs_to_remove_on_each_side <= clip_recordings_by_amount
            amount_of_epochs_to_remove_on_each_side = clip_recordings_by_amount

        for recording_wise_data_dict in self.recording_wise_data_list:
            recording_range_start, recording_range_stop = recording_wise_data_dict["epoch ids range"]
            number_of_epochs_in_recording = recording_range_stop - recording_range_start
            assert number_of_epochs_in_recording > 0
            recording_epochs_global_indices_list = list(range(recording_range_start, recording_range_stop))
            recording_epochs_labels_list = all_epochs_labels_list[recording_range_start:recording_range_stop]
            assert len(recording_epochs_global_indices_list) == len(recording_epochs_labels_list) == number_of_epochs_in_recording

            if amount_of_epochs_to_remove_on_each_side > 0:
                number_of_epochs_in_clipped_recording = number_of_epochs_in_recording - 2*amount_of_epochs_to_remove_on_each_side
                assert number_of_epochs_in_clipped_recording > 0
                recording_epochs_global_indices_list = recording_epochs_global_indices_list[amount_of_epochs_to_remove_on_each_side:-amount_of_epochs_to_remove_on_each_side]
                recording_epochs_labels_list = recording_epochs_labels_list[amount_of_epochs_to_remove_on_each_side:-amount_of_epochs_to_remove_on_each_side]
                assert len(recording_epochs_global_indices_list) == len(recording_epochs_labels_list) == number_of_epochs_in_clipped_recording

            self.dataset_epochs_global_indices_list += recording_epochs_global_indices_list
            self.dataset_epochs_labels_list += recording_epochs_labels_list

        self.dataset_length = len(self.dataset_epochs_global_indices_list)
        assert self.dataset_length == len(self.dataset_epochs_labels_list)

        if rebalance_set_by_oversampling:
            pre_oversampling_elements_per_class_counter = Counter(self.dataset_epochs_labels_list)
            assert sorted(list(pre_oversampling_elements_per_class_counter.keys())) == self.label_ids_list
            pre_oversampling_most_common_class_size = pre_oversampling_elements_per_class_counter.most_common(1)[0][1]

            dataset_epochs_global_indices_list = np.array(self.dataset_epochs_global_indices_list)
            dataset_epochs_global_indices_list = np.expand_dims(dataset_epochs_global_indices_list, axis=1)
            oversampler = RandomOverSampler(sampling_strategy="auto", random_state=random_seed)
            dataset_epochs_global_indices_list, self.dataset_epochs_labels_list = oversampler.fit_resample(dataset_epochs_global_indices_list, self.dataset_epochs_labels_list)

            dataset_epochs_global_indices_list = np.squeeze(dataset_epochs_global_indices_list)
            assert len(dataset_epochs_global_indices_list.shape) == 1
            self.dataset_epochs_global_indices_list = dataset_epochs_global_indices_list.tolist()

            self.dataset_length = len(self.dataset_epochs_global_indices_list)
            post_oversampling_elements_per_class_counter = Counter(self.dataset_epochs_labels_list)
            assert sorted(list(post_oversampling_elements_per_class_counter.keys())) == self.label_ids_list
            assert self.dataset_length == len(self.dataset_epochs_labels_list) == pre_oversampling_most_common_class_size * self.number_of_classes
            for value in post_oversampling_elements_per_class_counter.values():
                assert value == pre_oversampling_most_common_class_size

        self.__setup_done_flag = True

    def __len__(self):
        assert self.__setup_done_flag
        return self.dataset_length

    def __getitem__(self, item):
        assert self.__setup_done_flag
        assert item < self.__len__()
        output_dict = {}
        sampling_frequency = None

        central_epoch_global_index = self.dataset_epochs_global_indices_list[item]
        central_epoch_sequence_index = self.extra_epochs_on_each_side
        sequence_range_lower_bound = central_epoch_global_index - self.extra_epochs_on_each_side
        sequence_range_upper_bound = central_epoch_global_index + self.extra_epochs_on_each_side + 1
        sequence_global_indices = list(range(sequence_range_lower_bound, sequence_range_upper_bound))
        assert len(sequence_global_indices) == self.sequences_of_epochs_length
        assert sequence_global_indices[0] >= 0  # No negative indices

        sequence_of_epoch_labels_list = []
        sequence_of_epoch_matrices_list = []
        sequence_of_epoch_eeg_signals_list = []
        sequences_of_epoch_statistic_vectors_dict_of_lists = {}
        for statistic_vector in self.statistic_vectors_list:
            sequences_of_epoch_statistic_vectors_dict_of_lists[statistic_vector] = []

        recording_id = None
        for epoch_sequence_index in range(self.sequences_of_epochs_length):
            epoch_global_index = sequence_global_indices[epoch_sequence_index]
            epoch_data_dict = self.data_reader.get_element_data(epoch_global_index)

            epoch_label_id = epoch_data_dict["label id"]
            epoch_label = epoch_data_dict["label"]
            assert epoch_label in self.labels_list
            assert epoch_label_id == self.labels_list.index(epoch_label)
            if epoch_sequence_index == central_epoch_sequence_index:
                assert epoch_label_id == self.dataset_epochs_labels_list[item]

            sequence_of_epoch_labels_list.append(epoch_label_id)

            if recording_id is None:
                recording_id = epoch_data_dict["recording id"]
            assert recording_id == epoch_data_dict["recording id"]

            sequence_of_epoch_matrices_list.append(epoch_data_dict["matrices"])

            if self.transfer_eeg_epochs_flag:
                sequence_of_epoch_eeg_signals_list.append(epoch_data_dict["EEG signals"])
                if sampling_frequency is None:
                    sampling_frequency = epoch_data_dict["sampling frequency"]
                assert sampling_frequency == epoch_data_dict["sampling frequency"]

            if self.transfer_statistic_vectors_flag:
                for statistic_vectors in self.statistic_vectors_list:
                    sequences_of_epoch_statistic_vectors_dict_of_lists[statistic_vectors].append(epoch_data_dict["statistic vectors"][statistic_vectors])

        # shape (sequences_of_epochs_length,)
        sequence_of_epoch_labels_tensor = torch.Tensor(sequence_of_epoch_labels_list)
        assert sequence_of_epoch_labels_tensor.shape == (self.sequences_of_epochs_length,)

        # shape (number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, matrix_size, matrix_size)
        sequence_of_epoch_matrices_tensor = torch.stack(sequence_of_epoch_matrices_list, dim=1)
        assert len(sequence_of_epoch_matrices_tensor.shape) == 5
        assert sequence_of_epoch_matrices_tensor.shape[0] == self.number_of_channels
        assert sequence_of_epoch_matrices_tensor.shape[1] == self.sequences_of_epochs_length
        assert sequence_of_epoch_matrices_tensor.shape[3] == sequence_of_epoch_matrices_tensor.shape[4] == self.matrices_size

        output_dict["matrices"] = sequence_of_epoch_matrices_tensor

        # shape (number_of_channels, sequences_of_epochs_length, number_of_signals, number_of_subdivisions_per_epoch, subdivision_signal_length)
        # Here, (number_of_signals, number_of_subdivisions_per_epoch) == (matrix_size, number_of_matrices_per_epoch)
        if self.transfer_eeg_epochs_flag:
            sequence_of_epoch_signals_tensor = torch.stack(sequence_of_epoch_eeg_signals_list, dim=1)
            sequence_of_epoch_signals_tensor = sequence_of_epoch_signals_tensor.transpose(2, 3)  # Made a mistake in data saving format within preprocessor
            assert len(sequence_of_epoch_signals_tensor.shape) == 5
            assert sequence_of_epoch_signals_tensor.shape[0] == self.number_of_channels
            assert sequence_of_epoch_signals_tensor.shape[1] == self.sequences_of_epochs_length
            assert sequence_of_epoch_signals_tensor.shape[2] == self.matrices_size
            assert sequence_of_epoch_signals_tensor.shape[3] == sequence_of_epoch_matrices_tensor.shape[2]

            output_dict["EEG signals"] = sequence_of_epoch_signals_tensor
            output_dict["sampling frequency"] = sampling_frequency

        # shape (number_of_channels, sequences_of_epochs_length, number_of_vectors_per_epoch, vector_size)
        # Here, (number_of_vectors_per_epoch, vector_size) == (number_of_matrices_per_epoch, matrix_size)
        if self.transfer_statistic_vectors_flag:
            sequences_of_epoch_statistic_vectors_tensors_list = []
            for statistic_vector in self.statistic_vectors_list:
                sequence_of_epoch_statistic_vectors_tensor = torch.stack(sequences_of_epoch_statistic_vectors_dict_of_lists[statistic_vector], dim=1)
                assert len(sequence_of_epoch_statistic_vectors_tensor.shape) == 4
                assert sequence_of_epoch_statistic_vectors_tensor.shape[0] == self.number_of_channels
                assert sequence_of_epoch_statistic_vectors_tensor.shape[1] == self.sequences_of_epochs_length
                assert sequence_of_epoch_statistic_vectors_tensor.shape[2] == sequence_of_epoch_matrices_tensor.shape[2]
                assert sequence_of_epoch_statistic_vectors_tensor.shape[3] == self.matrices_size
                sequences_of_epoch_statistic_vectors_tensors_list.append(sequence_of_epoch_statistic_vectors_tensor)

            combined_sequences_of_epoch_statistic_vectors_tensor = torch.stack(sequences_of_epoch_statistic_vectors_tensors_list, dim=-1)
            output_dict["statistic matrices"] = combined_sequences_of_epoch_statistic_vectors_tensor

        if self.transfer_recording_wise_matrices_flag:

            recording_id_in_dataset = self.recording_indices.index(recording_id)
            recording_data_dict = self.recording_wise_data_list[recording_id_in_dataset]

            # shape (number_of_channels, matrix_size, matrix_size)
            recording_matrices = recording_data_dict["recording-wise matrices"]
            assert recording_matrices.shape == (self.number_of_channels, self.matrices_size, self.matrices_size)

            output_dict["recording-wise matrices"] = recording_matrices

            # shape (number_of_channels, vector_size)
            if self.transfer_statistic_vectors_flag:
                mean_statistic_vectors_list = []

                for statistic_vector in self.statistic_vectors_list:
                    mean_statistic_vectors = recording_data_dict["mean statistic vectors"][statistic_vector]
                    assert mean_statistic_vectors.shape == (self.number_of_channels, self.matrices_size)
                    mean_statistic_vectors_list.append(mean_statistic_vectors)

                combined_mean_statistic_vectors_tensor = torch.stack(mean_statistic_vectors_list, dim=-1)
                output_dict["recording mean statistic matrices"] = combined_mean_statistic_vectors_tensor

        return output_dict, sequence_of_epoch_labels_tensor



















