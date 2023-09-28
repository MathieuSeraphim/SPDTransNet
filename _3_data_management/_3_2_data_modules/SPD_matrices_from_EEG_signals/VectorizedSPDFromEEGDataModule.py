from copy import deepcopy
from os.path import dirname, realpath, join, isdir
import yaml
from typing import Union
from torch.utils.data import DataLoader
from _1_configs._1_z_miscellaneous.channel_wise_transformations.utils import get_channel_wise_transformations
from _3_data_management._3_2_data_modules.BaseDataModule import BaseDataModule
from _3_data_management._3_2_data_modules.DatasetWrapper import get_dataset_from_config
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.StandardVectorizedSPDFromEEGDataset\
    import StandardVectorizedSPDFromEEGDataset


class VectorizedSPDFromEEGDataModule(BaseDataModule):

    COMPATIBLE_DATASET_CLASSES = (StandardVectorizedSPDFromEEGDataset,)

    def __init__(self, dataset_config_file: str, batch_size: int, cross_validation_fold_index: int,
                 extra_epochs_on_each_side: int, signal_preprocessing_strategy: str,
                 channel_wise_transformations_config_file: str, channel_wise_transformations_as_hyphenated_list: str,
                 covariance_estimator: str, statistic_vectors_for_matrix_augmentation_as_hyphenated_list: str,
                 transfer_recording_wise_matrices: bool, rebalance_training_set_by_oversampling: bool = False,
                 clip_test_set_recordings_by_amount: Union[int, None] = None, dataloader_num_workers: int = 0,
                 no_covariances: bool = False, use_recording_wise_simple_covariances: bool = False,
                 get_epoch_eeg_signals: bool = False, random_seed: int = 42,
                 run_identifier: Union[str, int] = "default", augmentation_factor: float = 1.,
                 matrix_multiplication_factor: float = 1., svd_singular_value_minimum: Union[float, None] = None,
                 save_in_single_file: bool = False):
        r"""
        Constructor for the class, including arguments to initialize its utilized Dataset object.
        :param dataset_config_file: defines the EEG signals used, the labels and their order, and which DataReader (and
         thus Preprocessor) configuration file to use. Also contains the path to the utilized Dataset class, which must
         be in the COMPATIBLE_DATASET_CLASSES tuple (above).
        :param batch_size: self-explanatory. Doesn't have to be a power of 2.
        :param cross_validation_fold_index: one of the folds defined in the folder
         _1_z_miscellaneous/cross_validation_folds/[DATASET NAME] (with [DATASET NAME]) defined in the Preprocessor
         configuration file).
        :param extra_epochs_on_each_side: context epoch for the input sequence. Final sequence length will be
         2*extra_epochs_on_each_side + 1, with the central epoch being classified. The first and last
         extra_epochs_on_each_side epochs of each recording will not be classified.
        :param signal_preprocessing_strategy: the preprocessing strategy applied to the EEG signal (i.e. "raw_signals"
         or "z_score_normalization").
        :param channel_wise_transformations_config_file: the name of the .yaml file defining said transformation, within
         the folder _1_z_miscellaneous/channel_wise_transformations. Defines a dict, whose keys should not contain
         hyphens "-".
        :param channel_wise_transformations_as_hyphenated_list: keys for the aforementioned YAML-defined dict separated
         by hyphens "-". Order matters!
        :param covariance_estimator: i.e. "cov", "oas" or "mcd" (cf. Preprocessor).
        :param statistic_vectors_for_matrix_augmentation_as_hyphenated_list: keys defining statistic vectors (cf.
         Preprocessor), separated by hyphens "-". If multiple vectors are mentioned, they will be concatenated
          length-wise to build an augmentation matrix. Order matters!
        :param transfer_recording_wise_matrices: whether or not recording-wise matrices should be utilized. Controls the
         application of the whitening operation. Wording is a holdover from when the whitening was operated in the
         model (it is now done within the Dataset).
        :param rebalance_training_set_by_oversampling: if True, less populous classes will be randomly oversampled
         within the training set, so that their number is on par with that of the most populous class.
        :param clip_test_set_recordings_by_amount: ensures that the borders of recordings within the test set are
         clipped by at least the given amount. If extra_epochs_on_each_side > clip_test_set_recordings_by_amount, the
         behavior is unchanged. If clip_test_set_recordings_by_amount > extra_epochs_on_each_side, the first and last
         clip_test_set_recordings_by_amount epochs of each recording will not be classified for the test set only.
        :param dataloader_num_workers: the num_workers argument of the PyTorch DataLoader class.
        :param no_covariances: if set to True, non-diagonal elements of the pre-augmentation covariance matrices will be
         set to 0. If transfer_recording_wise_matrices is True, compute_recording_matrices_no_covariance must be True in
         the Preprocessor configuration.
        :param use_recording_wise_simple_covariances: only relevant if transfer_recording_wise_matrices is True. If
         use_recording_wise_simple_covariances is True, recording-wise covariance matrices are used for whitening.
         Otherwise, the affine invariant mean of the recordings' matrices are used. Set to False for best results.
        :param get_epoch_eeg_signals: if True, feeds EEG signals to the model. Very memory-consuming, especially with
         the data duplication resulting from our sequence-to-epoch strategy and our training set rebalancing.
        :param random_seed: used for oversampling and training set batch formation.
        :param run_identifier: used for log saving and checkpointing. Either set to "default", or read the relevant code
         to understand its behavior.
        :param augmentation_factor: used in matrix augmentation; controls the relative importance of the augmentation
         vector / matrix with regards to the original covariance matrix that it is used to augment.
        :param matrix_multiplication_factor: applied to all matrices after augmentation and/or whitening.
        :param svd_singular_value_minimum: during the SVD operation used to apply the matrix log for tokenization,
         should a singular value be below this threshold, it will be set to this threshold.
        :param save_in_single_file: if True, the tokenized matrices generated at initialization will be saved in a
         single .zip file (longer access time, lower number of inodes). Otherwise, they will be saved in one file per
         epoch (>58000 files for the entire dataset!).
        """
        self.save_hyperparameters(logger=False)
        super(VectorizedSPDFromEEGDataModule, self).__init__(batch_size, dataloader_num_workers, random_seed)

        self.__setup_done_flag = False

        dataset = get_dataset_from_config(dataset_config_file)
        assert isinstance(dataset, self.COMPATIBLE_DATASET_CLASSES)
        self.base_dataset = dataset
        self.dataset_name = dataset.data_reader.dataset_name

        statistic_vectors_for_matrix_augmentation\
            = statistic_vectors_for_matrix_augmentation_as_hyphenated_list.split("-")

        channel_wise_transformations_list = channel_wise_transformations_as_hyphenated_list.split("-")
        channel_wise_transformations = get_channel_wise_transformations(channel_wise_transformations_config_file,
                                                                        channel_wise_transformations_list)

        self.dataset_kwargs = {"extra_epochs_on_each_side": extra_epochs_on_each_side,
                               "signal_preprocessing_strategy": signal_preprocessing_strategy,
                               "channel_wise_transformations": channel_wise_transformations,
                               "covariance_estimator": covariance_estimator,
                               "statistic_vectors_for_matrix_augmentation": statistic_vectors_for_matrix_augmentation,
                               "transfer_recording_wise_matrices": transfer_recording_wise_matrices,
                               "no_covariances": no_covariances,
                               "use_recording_wise_simple_covariances": use_recording_wise_simple_covariances,
                               "get_epoch_eeg_signals": get_epoch_eeg_signals,
                               "random_seed": random_seed,
                               "augmentation_factor": augmentation_factor,
                               "matrix_multiplication_factor": matrix_multiplication_factor,
                               "svd_singular_value_minimum": svd_singular_value_minimum,
                               "save_in_single_file": save_in_single_file}
        self.rebalance_training_set_by_oversampling = rebalance_training_set_by_oversampling
        self.clip_test_set_recordings_by_amount = clip_test_set_recordings_by_amount

        self.run_identifier = run_identifier
        self.cross_validation_fold_index = cross_validation_fold_index

    def send_hparams_to_logger(self):
        hparams_dict = dict(self.hparams)

        datamodule_class_path = self.__module__ + "." + self.__class__.__name__
        datamodule_dict = {"class_path": datamodule_class_path, "init_args": hparams_dict}

        output_dict = {"datamodule": datamodule_dict}
        self.hparams.clear()
        self.hparams.update(output_dict)
        self.save_hyperparameters()

    def setup(self, stage: str):
        if not self.__setup_done_flag:
            self.send_hparams_to_logger()

            if isinstance(self.run_identifier, int):
                self.run_identifier = str(self.run_identifier)
            elif self.run_identifier == "default" and self.trainer is not None:
                self.run_identifier = str(self.trainer.logger.version)
            self.run_identifier = self.run_identifier + "_fold_%02d" % self.cross_validation_fold_index
            self.dataset_kwargs["run_identifier"] = self.run_identifier

            self.__setup_done_flag = True

    def get_cross_validation_recording_indices(self):
        current_script_directory = dirname(realpath(__file__))
        root_directory = dirname(dirname(dirname(current_script_directory)))
        configs_directory = join(root_directory, "_1_configs")
        miscellaneous_configs_directory = join(configs_directory, "_1_z_miscellaneous")
        cross_validation_folds_folder = join(miscellaneous_configs_directory, "cross_validation_folds")
        cross_validation_folds_for_dataset_folder = join(cross_validation_folds_folder, self.dataset_name)
        assert isdir(cross_validation_folds_for_dataset_folder)
        fold_filename = join(cross_validation_folds_for_dataset_folder, "fold_%s.yaml" % str(self.cross_validation_fold_index).zfill(2))
        with open(fold_filename, "r") as f:
            fold_dict = yaml.safe_load(f)
        return fold_dict

    def train_dataloader(self):
        if self.training_set is None:
            recording_indices_per_set_dict = self.get_cross_validation_recording_indices()
            self.training_set = deepcopy(self.base_dataset)
            self.training_set.setup(**self.dataset_kwargs, recording_indices=recording_indices_per_set_dict["training"],
                                    rebalance_set_by_oversampling=self.rebalance_training_set_by_oversampling,
                                    current_subset="training")
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.dataloader_num_workers, generator=self.generator)

    def val_dataloader(self):
        if self.validation_set is None:
            recording_indices_per_set_dict = self.get_cross_validation_recording_indices()
            self.validation_set = deepcopy(self.base_dataset)
            self.validation_set.setup(**self.dataset_kwargs,
                                      recording_indices=recording_indices_per_set_dict["validation"],
                                      current_subset="validation")
        return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        if self.test_set is None:
            recording_indices_per_set_dict = self.get_cross_validation_recording_indices()
            self.test_set = deepcopy(self.base_dataset)
            self.test_set.setup(**self.dataset_kwargs, recording_indices=recording_indices_per_set_dict["test"],
                                clip_recordings_by_amount=self.clip_test_set_recordings_by_amount,
                                current_subset="test")
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.dataloader_num_workers)






