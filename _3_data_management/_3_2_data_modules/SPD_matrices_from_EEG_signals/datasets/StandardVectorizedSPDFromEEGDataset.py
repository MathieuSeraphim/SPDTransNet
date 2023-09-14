import pickle
from shutil import rmtree
import numcodecs
import zarr
from os import mkdir, listdir, remove
from os.path import dirname, realpath, join, isdir, isfile
import torch
from typing import List, Tuple, Any, Union
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.non_spd_datasets.SPDFromEEGDataset import \
    SPDFromEEGDataset
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.layers.MatrixAugmentationLayer import \
    MatrixAugmentationLayer
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.layers.MatrixVectorizationLayer import \
    MatrixVectorizationLayer
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.layers.MatrixWhiteningLayer import \
    MatrixWhiteningLayer


class StandardVectorizedSPDFromEEGDataset(SPDFromEEGDataset):

    EXTRA_DIMENSIONS_COMPARED_TO_SINGLE_MATRICES = 1

    def __init__(self, eeg_signals: List[str], labels: List[str], data_reader_config_file: str):
        super().__init__(eeg_signals, labels, data_reader_config_file)

        self.augmentation_layer = MatrixAugmentationLayer()
        self.whitening_layer = MatrixWhiteningLayer()
        self.vectorization_layer = MatrixVectorizationLayer()

        current_script_directory = dirname(realpath(__file__))
        root_directory = dirname(dirname(dirname(dirname(current_script_directory))))
        self.temporary_vectorized_data_storage_folder = join(root_directory, "tmp_vectorized_data")
        if not isdir(self.temporary_vectorized_data_storage_folder):
            try:
                mkdir(self.temporary_vectorized_data_storage_folder)
            except FileExistsError:
                pass

        self.save_in_single_file = True
        self.augmented_matrices_size = -1
        self.vectorized_matrices_size = -1
        self.temporary_vectorized_data_storage_location = None
        self.current_subset = None

    def __del__(self):
        if self._SPDFromEEGDataset__setup_done_flag:
            self.delete_tmp_storage()
            if isfile(self.temporary_vectorized_data_storage_folder):
                if len(listdir(self.temporary_vectorized_data_storage_folder)) == 0:
                    remove(self.temporary_vectorized_data_storage_folder)

    def delete_tmp_storage(self):
        if isfile(self.temporary_vectorized_data_storage_location):
            remove(self.temporary_vectorized_data_storage_location)
        elif isdir(self.temporary_vectorized_data_storage_location):
            rmtree(self.temporary_vectorized_data_storage_location)

    def setup(self, recording_indices: List[int], extra_epochs_on_each_side: int, signal_preprocessing_strategy: str,
              channel_wise_transformations: List[Tuple[str, Any]], covariance_estimator: str,
              statistic_vectors_for_matrix_augmentation: List[str], transfer_recording_wise_matrices: bool,
              rebalance_set_by_oversampling: bool = False, clip_recordings_by_amount: Union[int, None] = None,
              use_recording_wise_simple_covariances: bool = False, no_covariances: bool = False,
              get_epoch_eeg_signals: bool = False, random_seed: int = 42, run_identifier: str = "default",
              current_subset: str = "none", augmentation_factor: float = 1., matrix_multiplication_factor: float = 1.,
              svd_singular_value_minimum: Union[float, None] = None, save_in_single_file: bool = False):

        self.save_in_single_file = save_in_single_file

        self.temporary_vectorized_data_storage_location = join(self.temporary_vectorized_data_storage_folder,
                                                               "run_%s_set_%s" % (run_identifier, current_subset))
        if self.save_in_single_file:
            self.temporary_vectorized_data_storage_location += ".zip"

        self.delete_tmp_storage()  # Removing data from previous runs, if any
        if self.save_in_single_file:
            with zarr.ZipStore(path=self.temporary_vectorized_data_storage_location) as storage_structure:
                zarr.group(store=storage_structure)
        else:
            mkdir(self.temporary_vectorized_data_storage_location)

        super(StandardVectorizedSPDFromEEGDataset, self).setup(recording_indices, extra_epochs_on_each_side,
                                                               signal_preprocessing_strategy,
                                                               channel_wise_transformations, covariance_estimator,
                                                               statistic_vectors_for_matrix_augmentation,
                                                               transfer_recording_wise_matrices,
                                                               rebalance_set_by_oversampling, clip_recordings_by_amount,
                                                               use_recording_wise_simple_covariances, no_covariances,
                                                               get_epoch_eeg_signals, random_seed)

        augmentation_layer_kwargs = {
            "matrix_size": self.matrices_size,
            "augmentation_size": len(self.statistic_vectors_list),
            "initial_augmentation_factor": augmentation_factor,
            "augmentation_factor_learnable": False
        }
        matrix_augmentation_active_flag, self.augmented_matrices_size\
            = self.augmentation_layer.setup(**augmentation_layer_kwargs)
        assert matrix_augmentation_active_flag == self.transfer_statistic_vectors_flag

        whitening_layer_kwargs = {
            "matrix_size": self.augmented_matrices_size,
            "operate_whitening": self.transfer_recording_wise_matrices_flag,
            "extra_dimensions": self.EXTRA_DIMENSIONS_COMPARED_TO_SINGLE_MATRICES,
            "matrix_multiplication_factor": matrix_multiplication_factor
        }
        self.whitening_layer.setup(**whitening_layer_kwargs)

        vectorization_layer_kwargs = {
            "matrix_size": self.augmented_matrices_size,
            "svd_singular_value_minimum": svd_singular_value_minimum
        }
        self.vectorized_matrices_size = self.vectorization_layer.setup(**vectorization_layer_kwargs)

        print()
        print("Generating data in file %s..." % self.temporary_vectorized_data_storage_location)

        for recording_wise_data_dict in self.recording_wise_data_list:

            storage_structure = None
            if self.save_in_single_file:
                storage_structure = zarr.ZipStore(path=self.temporary_vectorized_data_storage_location)

            recording_range_start, recording_range_stop = recording_wise_data_dict["epoch ids range"]
            recording_epochs_global_indices_list = list(range(recording_range_start, recording_range_stop))

            recording_wise_matrices = None
            if self.transfer_recording_wise_matrices_flag:
                recording_wise_matrices = recording_wise_data_dict["recording-wise matrices"]
                assert recording_wise_matrices.shape == (self.number_of_channels, self.matrices_size, self.matrices_size)

                if self.transfer_statistic_vectors_flag:
                    mean_statistic_vectors_list = []
                    for statistic_vector in self.statistic_vectors_list:
                        mean_statistic_vectors = recording_wise_data_dict["mean statistic vectors"][statistic_vector]
                        assert mean_statistic_vectors.shape == (self.number_of_channels, self.matrices_size)
                        mean_statistic_vectors_list.append(mean_statistic_vectors)

                    combined_mean_statistic_vectors_tensor = torch.stack(mean_statistic_vectors_list, dim=-1)
                    recording_wise_matrices = self.augmentation_layer(recording_wise_matrices,
                                                                      combined_mean_statistic_vectors_tensor)

            for epoch_global_index in recording_epochs_global_indices_list:
                original_epoch_data_dict = self.data_reader.get_element_data(epoch_global_index)
                list_of_keys_to_transfer = list(original_epoch_data_dict.keys())

                epoch_matrices = original_epoch_data_dict["matrices"]
                assert len(epoch_matrices.shape) == 4
                assert epoch_matrices.shape[0] == self.number_of_channels
                assert epoch_matrices.shape[2] == epoch_matrices.shape[3] == self.matrices_size
                number_of_matrices_per_epoch = epoch_matrices.shape[1]
                list_of_keys_to_transfer.remove("matrices")

                if self.transfer_statistic_vectors_flag:
                    epoch_statistic_vectors_list = []
                    for statistic_vector in self.statistic_vectors_list:
                        epoch_statistic_vectors = original_epoch_data_dict["statistic vectors"][statistic_vector]
                        assert epoch_statistic_vectors.shape == (self.number_of_channels, number_of_matrices_per_epoch,
                                                                 self.matrices_size)
                        epoch_statistic_vectors_list.append(epoch_statistic_vectors)

                    combined_epoch_statistic_vectors_tensor = torch.stack(epoch_statistic_vectors_list, dim=-1)
                    epoch_matrices = self.augmentation_layer(epoch_matrices, combined_epoch_statistic_vectors_tensor)

                    list_of_keys_to_transfer.remove("statistic vectors")

                if self.transfer_recording_wise_matrices_flag:
                    epoch_matrices = self.whitening_layer(epoch_matrices, recording_wise_matrices)

                # (number_of_channels, number_of_matrices_per_epoch, vector_size)
                epoch_vectorized_matrices = self.vectorization_layer(epoch_matrices)
                assert epoch_vectorized_matrices.shape == (self.number_of_channels, number_of_matrices_per_epoch, self.vectorized_matrices_size)

                modified_data_dict = {"vectorized matrices": epoch_vectorized_matrices}
                for key in list_of_keys_to_transfer:
                    modified_data_dict[key] = original_epoch_data_dict[key]

                if self.save_in_single_file:
                    data_storage = zarr.open_group(store=storage_structure)
                    data_storage.array(name=str(epoch_global_index), data=modified_data_dict, dtype=object,
                                       object_codec=numcodecs.Pickle(), compressor=None)
                else:
                    epoch_tmp_filename = join(self.temporary_vectorized_data_storage_location,
                                              str(epoch_global_index) + ".pkl")
                    with open(epoch_tmp_filename, 'wb') as f:
                        pickle.dump(modified_data_dict, f)

            if self.save_in_single_file:
                storage_structure.close()

    def __getitem__(self, item):
        assert self._SPDFromEEGDataset__setup_done_flag
        assert item < self.__len__()
        output_dict = {}
        sampling_frequency = None

        storage_structure = None
        if self.save_in_single_file:
            storage_structure = zarr.ZipStore(path=self.temporary_vectorized_data_storage_location, mode="r",
                                              compression=0)

        central_epoch_global_index = self.dataset_epochs_global_indices_list[item]
        central_epoch_sequence_index = self.extra_epochs_on_each_side
        sequence_range_lower_bound = central_epoch_global_index - self.extra_epochs_on_each_side
        sequence_range_upper_bound = central_epoch_global_index + self.extra_epochs_on_each_side + 1
        sequence_global_indices = list(range(sequence_range_lower_bound, sequence_range_upper_bound))
        assert len(sequence_global_indices) == self.sequences_of_epochs_length
        assert sequence_global_indices[0] >= 0  # No negative indices

        sequence_of_epoch_labels_list = []
        sequence_of_epoch_vectorized_matrices_list = []
        sequence_of_epoch_eeg_signals_list = []

        recording_id = None

        for epoch_sequence_index in range(self.sequences_of_epochs_length):
            epoch_global_index = sequence_global_indices[epoch_sequence_index]

            if self.save_in_single_file:
                epoch_data_dict = zarr.load(store=storage_structure, path=str(epoch_global_index)).item()
            else:
                epoch_data_filename = join(self.temporary_vectorized_data_storage_location,
                                           str(epoch_global_index) + ".pkl")
                with open(epoch_data_filename, "rb") as f:
                    epoch_data_dict = pickle.load(f)

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

            sequence_of_epoch_vectorized_matrices_list.append(torch.tensor(epoch_data_dict["vectorized matrices"]))

            if self.transfer_eeg_epochs_flag:
                sequence_of_epoch_eeg_signals_list.append(torch.tensor(epoch_data_dict["EEG signals"]))
                if sampling_frequency is None:
                    sampling_frequency = epoch_data_dict["sampling frequency"]
                assert sampling_frequency == epoch_data_dict["sampling frequency"]

        # shape (sequences_of_epochs_length,)
        sequence_of_epoch_labels_tensor = torch.Tensor(sequence_of_epoch_labels_list)
        assert sequence_of_epoch_labels_tensor.shape == (self.sequences_of_epochs_length,)

        # shape (number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, vector_size)
        sequence_of_epoch_vectorized_matrices_tensor = torch.stack(sequence_of_epoch_vectorized_matrices_list, dim=1)
        assert len(sequence_of_epoch_vectorized_matrices_tensor.shape) == 4
        assert sequence_of_epoch_vectorized_matrices_tensor.shape[0] == self.number_of_channels
        assert sequence_of_epoch_vectorized_matrices_tensor.shape[1] == self.sequences_of_epochs_length
        assert sequence_of_epoch_vectorized_matrices_tensor.shape[3] == self.vectorized_matrices_size

        output_dict["vectorized matrices"] = sequence_of_epoch_vectorized_matrices_tensor

        # shape (number_of_channels, sequences_of_epochs_length, number_of_signals, number_of_subdivisions_per_epoch, subdivision_signal_length)
        # Here, (number_of_signals, number_of_subdivisions_per_epoch) == (matrix_size, number_of_matrices_per_epoch)
        if self.transfer_eeg_epochs_flag:
            sequence_of_epoch_signals_tensor = torch.stack(sequence_of_epoch_eeg_signals_list, dim=1)
            sequence_of_epoch_signals_tensor = sequence_of_epoch_signals_tensor.transpose(2, 3)  # Made a mistake in data saving format within preprocessor
            assert len(sequence_of_epoch_signals_tensor.shape) == 5
            assert sequence_of_epoch_signals_tensor.shape[0] == self.number_of_channels
            assert sequence_of_epoch_signals_tensor.shape[1] == self.sequences_of_epochs_length
            assert sequence_of_epoch_signals_tensor.shape[2] == self.matrices_size
            assert sequence_of_epoch_signals_tensor.shape[3] == sequence_of_epoch_vectorized_matrices_tensor.shape[2]

            output_dict["EEG signals"] = sequence_of_epoch_signals_tensor
            output_dict["sampling frequency"] = sampling_frequency

        return output_dict, sequence_of_epoch_labels_tensor



















