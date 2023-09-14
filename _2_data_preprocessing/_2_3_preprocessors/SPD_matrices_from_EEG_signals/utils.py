import numpy as np
from typing import Any, List, Union, Tuple, Dict
from scipy.signal import butter, lfilter, periodogram
from scipy.stats import zscore
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.test import is_sym_pos_def


# signal of shape (signal_length,) or (number_of_signals, signal_length)
SIGNAL_PREPROCESSING_STRATEGIES = ["raw_signals", "z_score_normalization"]
def signal_preprocessing(signal: np.ndarray, strategy: str, **kwargs):
    assert 0 < len(signal.shape) <= 2
    assert strategy in SIGNAL_PREPROCESSING_STRATEGIES
    if strategy == "raw_signals":
        return signal
    elif strategy == "z_score_normalization":
        return signal_z_score_normalization(signal)
    else:
        raise NotImplementedError


def signal_z_score_normalization(signal: np.ndarray):
    return zscore(signal, axis=-1)


# signal of shape (signal_length,) or (number_of_signals, signal_length)
CHANNEL_TRANSFORMATIONS = ["none", "bandpass_filtering"]
def channel_wise_signal_transformation(signal: np.ndarray, channel_transformation_type: str,
                                       channel_transformation_config: Any, **kwargs):
    assert 0 < len(signal.shape) <= 2
    assert channel_transformation_type in CHANNEL_TRANSFORMATIONS
    if channel_transformation_type == "none":
        assert channel_transformation_config is None
        return signal
    elif channel_transformation_type == "bandpass_filtering":
        return signal_bandpass_filtering(signal, channel_transformation_config, **kwargs)


DEFAULT_BANDPASS_FILTER_ORDER = 4
def signal_bandpass_filtering(signal: np.ndarray, filter_config: List, sampling_frequency: float):
    assert len(filter_config) >= 2

    lowcut = filter_config[0]
    highcut = filter_config[1]

    if len(filter_config) == 2:
        order = DEFAULT_BANDPASS_FILTER_ORDER
    elif len(filter_config) == 3:
        order = filter_config[2]
    else:
        raise NotImplementedError

    Wn = np.array([lowcut, highcut]) / (sampling_frequency / 2)
    b, a = butter(order, Wn, "bandpass")
    return lfilter(b, a, signal, axis=-1)


# signal of shape (..., signal_length)
# output of shape (..., number_of_windows, window_size_in_steps)
# or (..., number_of_sequences, length_of_sequences, window_size_in_steps)
def subdivide_signal(signal: np.ndarray, window_size_in_steps: int, length_of_sequences_if_any: Union[int, None]):
    signal_length = signal.shape[-1]
    total_number_of_windows = int(signal_length / window_size_in_steps)
    assert total_number_of_windows == signal_length / window_size_in_steps  # Making sure it's an integer

    windowed_signal = np.stack(np.split(signal, total_number_of_windows, axis=-1), axis=-2)

    if length_of_sequences_if_any is None:
        return windowed_signal

    total_number_of_sequences = int(total_number_of_windows / length_of_sequences_if_any)
    assert total_number_of_sequences == total_number_of_windows / length_of_sequences_if_any  # Making sure it's an integer

    return windowed_signal.reshape(-1, total_number_of_sequences, length_of_sequences_if_any, window_size_in_steps)


# signal of shape (number_of_signals, number_of_sequences, length_of_sequences, window_size_in_steps)
# or (number_of_signals, number_of_windows, window_size_in_steps)
# output_signal of shape (total_number_of_windows, number_of_signals, window_size_in_steps)
# number_of_windows_tuple of shape (number_of_sequences, length_of_sequences) or (number_of_windows,)
def batch_windowed_signal_reformatting(signal: np.ndarray):
    assert 3 <= len(signal.shape) <= 4
    
    if len(signal.shape) == 4:
        number_of_signals, number_of_sequences, length_of_sequences, window_size_in_steps = signal.shape
        total_number_of_windows = number_of_sequences * length_of_sequences
        number_of_windows_tuple = (number_of_sequences, length_of_sequences)
        signal = signal.reshape(number_of_signals, total_number_of_windows, window_size_in_steps)
    else:
        assert len(signal.shape) == 3
        number_of_signals, number_of_windows, window_size_in_steps = signal.shape
        total_number_of_windows = number_of_windows
        number_of_windows_tuple = (number_of_windows,)

    # (total_number_of_windows, number_of_signals, window_size_in_steps)
    output_signal = signal.transpose(1, 0, 2)
    assert output_signal.shape == (total_number_of_windows, number_of_signals, window_size_in_steps)
    
    return output_signal, number_of_windows_tuple


# matrices of shape (total_number_of_matrices, matrix_size, matrix_size)
# output of shape (number_of_sequences, length_of_sequences, matrix_size, matrix_size)
# or (number_of_matrices, matrix_size, matrix_size)
def batch_covariance_matrices_reformatting(matrices: np.ndarray, number_of_matrices_tuple: Tuple):
    assert len(matrices.shape) == 3
    total_number_of_matrices, matrix_size, matrix_size_2 = matrices.shape
    assert matrix_size == matrix_size_2
    
    assert 1 <= len(number_of_matrices_tuple) <= 2
    if len(number_of_matrices_tuple) == 1:
        assert total_number_of_matrices == number_of_matrices_tuple[0]
        return matrices  # No transformation

    number_of_sequences, length_of_sequences = number_of_matrices_tuple
    assert total_number_of_matrices == number_of_sequences * length_of_sequences
    return matrices.reshape(number_of_sequences, length_of_sequences, matrix_size, matrix_size)


# vector of shape (total_number_of_vectors, vector_size)
# output of shape (number_of_sequences, length_of_sequences, vector_size)
# or (number_of_vectors, vector_size)
def batch_statistic_vectors_reformatting(vectors: np.ndarray, number_of_vectors_tuple: Tuple):
    assert len(vectors.shape) == 2
    total_number_of_vectors, vector_size = vectors.shape

    assert 1 <= len(number_of_vectors_tuple) <= 2
    if len(number_of_vectors_tuple) == 1:
        assert total_number_of_vectors == number_of_vectors_tuple[0]
        return vectors  # No transformation

    number_of_sequences, length_of_sequences = number_of_vectors_tuple
    assert total_number_of_vectors == number_of_sequences * length_of_sequences
    return vectors.reshape(number_of_sequences, length_of_sequences, vector_size)


# signal of shape (total_number_of_windows, number_of_signals, window_size_in_steps)
# number_of_matrices_tuple of shape (number_of_sequences, length_of_sequences) or (number_of_windows,)
# output of shape (number_of_sequences, length_of_sequences, number_of_signals, number_of_signals)
# or (number_of_windows, number_of_signals, number_of_signals)
ESTIMATORS = ["cov", "mcd", "oas"]
def batch_windowed_signal_to_covariance_matrices(signal: np.ndarray, number_of_matrices_tuple: Tuple,
                                                 covariance_estimator: str,
                                                 estimator_extra_args: Union[Dict[str, Any], None] = None):
    assert covariance_estimator in ESTIMATORS
    estimator_kwargs = {}
    if covariance_estimator == "mcd":
        assert "random_state" in estimator_extra_args
        estimator_kwargs["random_state"] = estimator_extra_args["random_state"]

    assert len(signal.shape) == 3
    total_number_of_windows, number_of_signals, window_size_in_steps = signal.shape

    covariance_matrices = covariances(signal, estimator=covariance_estimator, **estimator_kwargs)

    assert is_sym_pos_def(covariance_matrices)
    assert np.isreal(covariance_matrices).all()
    assert len(covariance_matrices.shape) == 3
    total_number_of_matrices, matrix_dim_1, matrix_dim_2 = covariance_matrices.shape
    assert (total_number_of_matrices, matrix_dim_1, matrix_dim_2)\
           == (total_number_of_windows, number_of_signals, number_of_signals)
    
    return batch_covariance_matrices_reformatting(covariance_matrices, number_of_matrices_tuple)


# signal of shape (number_of_signals, window_size_in_steps)
# output of shape (number_of_signals, number_of_signals)
def single_window_signal_to_covariance_matrices(signal: np.ndarray, covariance_estimator: str,
                                                estimator_extra_args: Union[Dict[str, Any], None] = None):
    signal_shape = signal.shape
    assert len(signal_shape) == 2
    signal = np.expand_dims(signal, axis=0)
    number_of_matrices_tuple = (1,)
    single_covariance_matrix = batch_windowed_signal_to_covariance_matrices(signal, number_of_matrices_tuple,
                                                                            covariance_estimator, estimator_extra_args)
    assert single_covariance_matrix.shape == (1, signal_shape[0], signal_shape[0])
    return np.squeeze(single_covariance_matrix, axis=0)


# signal of shape (total_number_of_windows, number_of_signals, window_size_in_steps)
# number_of_vectors_tuple of shape (number_of_sequences, length_of_sequences) or (number_of_windows,)
# output of shape (number_of_sequences, length_of_sequences, number_of_signals)
# or (_number_of_windows, number_of_signals)
STATISTICS = ["psd", "mean", "max_minus_min"]
def batch_windowed_signal_to_statistic_vectors(signal: np.ndarray, number_of_vectors_tuple: Tuple, statistic: str,
                                               **kwargs):
    assert statistic in STATISTICS
    assert len(signal.shape) == 3
    total_number_of_windows, number_of_signals, window_size_in_steps = signal.shape

    if statistic == "mean":
        statistic_vectors = batch_windowed_signal_to_mean_vectors_computation(signal)
    elif statistic == "max_minus_min":
        statistic_vectors = batch_windowed_signal_to_amplitude_differential_vectors_computation(signal)
    elif statistic == "psd":
        statistic_vectors = batch_windowed_signal_to_power_spectral_density_vectors_computation(signal, **kwargs)
    else:
        raise NotImplementedError

    assert np.isreal(statistic_vectors).all()
    assert len(statistic_vectors.shape) == 2
    total_number_of_vectors, vector_size = statistic_vectors.shape
    assert (total_number_of_vectors, vector_size) == (total_number_of_windows, number_of_signals)

    return batch_statistic_vectors_reformatting(statistic_vectors, number_of_vectors_tuple)


# signal of shape (number_of_signals, total_number_of_windows, window_size_in_steps)
# output of shape (number_of_signals, total_number_of_windows)
def batch_windowed_signal_to_mean_vectors_computation(signal: np.ndarray):
    return signal.mean(axis=-1)


# signal of shape (number_of_signals, total_number_of_windows, window_size_in_steps)
# output of shape (number_of_signals, total_number_of_windows)
def batch_windowed_signal_to_amplitude_differential_vectors_computation(signal: np.ndarray):
    signal_max = signal.max(axis=-1)
    signal_min = signal.min(axis=-1)
    return signal_max - signal_min


# signal of shape (number_of_signals, total_number_of_windows, window_size_in_steps)
# output of shape (number_of_signals, total_number_of_windows)
def batch_windowed_signal_to_power_spectral_density_vectors_computation(signal: np.ndarray, sampling_frequency: float):
    _, signal_periodograms = periodogram(signal, sampling_frequency, axis=-1)
    signal_mean_psd = signal_periodograms.mean(axis=-1)
    return signal_mean_psd


# matrices of shape (..., matrix_size, matrix_size)
def batch_remove_non_diagonal_elements_from_matrices(matrices: np.ndarray):
    assert len(matrices.shape) >= 2
    matrix_size, matrix_size_2 = matrices.shape[-2:]

    identity = np.identity(matrix_size)
    identity_expanded = np.broadcast_to(identity, matrices.shape)
    return np.multiply(matrices, identity_expanded)


# matrices of shape (..., matrix_size, matrix_size)
# output of shape (matrix_size, matrix_size)
def batch_spd_matrices_affine_invariant_mean(matrices: np.ndarray, remove_non_diagonal_values: bool = False):
    assert len(matrices.shape) >= 2
    matrix_size, matrix_size_2 = matrices.shape[-2:]
    assert matrix_size == matrix_size_2

    matrices = matrices.reshape(-1, matrix_size, matrix_size)
    if remove_non_diagonal_values:
        matrices = batch_remove_non_diagonal_elements_from_matrices(matrices)

    assert is_sym_pos_def(matrices)
    assert np.isreal(matrices).all()

    mean_matrix = mean_riemann(matrices)
    assert mean_matrix.shape == (matrix_size, matrix_size)
    assert is_sym_pos_def(mean_matrix)
    assert np.isreal(mean_matrix).all()

    return mean_matrix


# vectors of shape (..., vector_size)
# output of shape (vector_size,)
def batch_vectors_euclidean_mean(vectors: np.ndarray):
    assert len(vectors.shape) >= 1
    vector_size = vectors.shape[-1]

    vectors = vectors.reshape(-1, vector_size)
    assert np.isreal(vectors).all()

    mean_vector = vectors.mean(axis=0)
    assert mean_vector.shape == (vector_size,)
    assert np.isreal(mean_vector).all()

    return mean_vector







    
