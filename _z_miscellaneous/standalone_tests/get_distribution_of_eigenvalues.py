from os import makedirs
from os.path import dirname, realpath, join, isdir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.nn import Flatten
import torch
from _3_data_management._3_2_data_modules.SPD_matrices_from_EEG_signals.datasets.non_spd_datasets.SPDFromEEGDataset import \
    SPDFromEEGDataset
from _4_models.utils import matrix_pow

eeg_signals = ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2"]
labels = ["N3", "N2", "N1", "REM", "Awake"]
extra_epochs_on_each_side = 0
transfer_recording_wise_matrices = True
data_reader_config_file = "SPD_matrices_from_EEG_config.yaml"
preprocessed_dataset_name = "MASS_SS3_dataset_with_EUSIPCO_signals_config_backup"
clip_recordings_by_amount = None
rebalance_set_by_oversampling = False
recording_indices = list(range(62))
recording_indices_full_list = [[i] for i in recording_indices] + [recording_indices]
channel_wise_transformations = [["none", None],
                                ["bandpass_filtering", [0.5, 4]],
                                ["bandpass_filtering", [4, 8]],
                                ["bandpass_filtering", [8, 13]],
                                ["bandpass_filtering", [13, 22]],
                                ["bandpass_filtering", [22, 30]],
                                ["bandpass_filtering", [30, 45]]]
signal_preprocessing_strategy = "z_score_normalization"
statistic_vectors_to_return = []
no_covariances = False

covariance_estimators_list = ["cov", "oas"]
use_recording_wise_simple_covariances_list = [True, False]

number_of_channels = len(channel_wise_transformations)
matrix_size = len(eeg_signals)
matrices_per_epoch = 30

upper_percentile = 95
multipliers = [1, 100]

save_recording_wise_distribution = True

flatten = Flatten()
current_script_directory = dirname(realpath(__file__))

for recording_indices in recording_indices_full_list:
    for multiplier in multipliers:
        output_folder = join(current_script_directory, "get_distribution_of_eigenvalues_times_%d" % multiplier)
    
        unwhitened_matrices_already_processed = {}
        for key in covariance_estimators_list:
            unwhitened_matrices_already_processed[key] = False
    
        for covariance_estimator in covariance_estimators_list:
            for use_recording_wise_simple_covariances in use_recording_wise_simple_covariances_list:
    
                dataset = SPDFromEEGDataset(eeg_signals, labels, data_reader_config_file)
                dataset.setup(recording_indices, extra_epochs_on_each_side, signal_preprocessing_strategy,
                              channel_wise_transformations, covariance_estimator, statistic_vectors_to_return,
                              transfer_recording_wise_matrices, rebalance_set_by_oversampling,
                              clip_recordings_by_amount,
                              use_recording_wise_simple_covariances=use_recording_wise_simple_covariances,
                              no_covariances=no_covariances)
    
                unwhitened_matrices_already_processed_flag = unwhitened_matrices_already_processed[covariance_estimator]
                if not unwhitened_matrices_already_processed_flag:
                    unwhitened_matrices_already_processed[covariance_estimator] = True
    
                accumulated_unwhitened_matrices_list = []
                accumulated_whitened_matrices_list = []
                for i in range(len(dataset)):
                    data, _ = dataset[i]
                    matrices = data["matrices"]
                    whitening_matrices = data["recording-wise matrices"]
    
                    assert matrices.shape == (number_of_channels, 1, matrices_per_epoch, matrix_size, matrix_size)
                    assert whitening_matrices.shape == (number_of_channels, matrix_size, matrix_size)
    
                    matrices = matrices.squeeze(1)
    
                    if not unwhitened_matrices_already_processed_flag:
                        accumulated_unwhitened_matrices_list.append(matrices)
    
                    whitening_matrices = matrix_pow(whitening_matrices, -.5)
                    whitening_matrices = whitening_matrices.unsqueeze(-3)
                    whitened_matrices = torch.matmul(whitening_matrices, torch.matmul(matrices, whitening_matrices))
    
                    assert len(whitening_matrices.shape) == len(matrices.shape) == len(whitened_matrices.shape) == 4
                    accumulated_whitened_matrices_list.append(whitened_matrices)
    
                accumulated_whitened_matrices = torch.stack(accumulated_whitened_matrices_list, dim=1)
                whitened_matrices_eigenvalues = torch.linalg.eigvalsh(accumulated_whitened_matrices)
                whitened_matrices_eigenvalues = flatten(whitened_matrices_eigenvalues).numpy() * multiplier
    
                log_whitened_matrices_eigenvalues = np.log10(whitened_matrices_eigenvalues)
    
                if not unwhitened_matrices_already_processed_flag:
                    accumulated_unwhitened_matrices = torch.stack(accumulated_unwhitened_matrices_list, dim=1)
                    unwhitened_matrices_eigenvalues = torch.linalg.eigvalsh(accumulated_unwhitened_matrices)
                    unwhitened_matrices_eigenvalues = flatten(unwhitened_matrices_eigenvalues).numpy() * multiplier
    
                    log_unwhitened_matrices_eigenvalues = np.log10(unwhitened_matrices_eigenvalues)
    
                for channel_id in range(number_of_channels):
                    channel_folder = join(output_folder, "channel_%d" % channel_id)
                    final_folder = join(channel_folder, "estimator_%s" % covariance_estimator)
                    
                    if len(recording_indices) == 1:
                        final_folder = join(final_folder, "recording_wise")
                        final_folder = join(final_folder, "id_%02d" % recording_indices[0])
                    
                    if not isdir(final_folder):
                        makedirs(final_folder)
    
                    if use_recording_wise_simple_covariances:
                        whitened_filename = "whitened_using_simple_recording_wise_covariance_matrices"
                    else:
                        whitened_filename = "whitened_using_affine_invariant_average_of_recording_matrices"
    
                    log_whitened_filename = join(final_folder, "log_" + whitened_filename + ".png")
                    whitened_filename = join(final_folder, whitened_filename + ".png")
    
                    plt.close("all")
    
                    lower_bound = np.min(whitened_matrices_eigenvalues[channel_id])
                    upper_bound = np.percentile(whitened_matrices_eigenvalues[channel_id], upper_percentile)
                    distribution_plot = sns.displot(whitened_matrices_eigenvalues[channel_id], binrange=(lower_bound, upper_bound))
                    distribution_plot.savefig(whitened_filename)
    
                    distribution_plot = sns.displot(log_whitened_matrices_eigenvalues[channel_id])
                    distribution_plot.savefig(log_whitened_filename)
    
                    if not unwhitened_matrices_already_processed_flag:
                        unwhitened_filename = join(final_folder, "unwhitened.png")
                        log_unwhitened_filename = join(final_folder, "log_unwhitened.png")
    
                        lower_bound = np.min(unwhitened_matrices_eigenvalues[channel_id])
                        upper_bound = np.percentile(unwhitened_matrices_eigenvalues[channel_id], upper_percentile)
                        distribution_plot = sns.displot(unwhitened_matrices_eigenvalues[channel_id], binrange=(lower_bound, upper_bound))
                        distribution_plot.savefig(unwhitened_filename)
    
                        distribution_plot = sns.displot(log_unwhitened_matrices_eigenvalues[channel_id])
                        distribution_plot.savefig(log_unwhitened_filename)





