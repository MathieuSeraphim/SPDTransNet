from typing import Union
import torch
from torch.nn import Linear
from _4_models._4_1_sequence_based_models.data_formatting_block.BaseDataFormattingBlock import BaseDataFormattingBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.VectorChannelWiseCombinationBySuccessionLayer import \
    VectorChannelWiseCombinationBySuccessionLayer


class VectorizedSPDFromEEGDataMultichannelSuccessionReformattingBlock(BaseDataFormattingBlock):

    INDEX_OF_CHANNEL_DIMENSION = 1

    def __init__(self):
        super(VectorizedSPDFromEEGDataMultichannelSuccessionReformattingBlock, self).__init__()
        self.__setup_done_flag = False

        self.combination_layer = VectorChannelWiseCombinationBySuccessionLayer()
        self.final_linear_projection_layer = None

        self.number_of_vectors_per_epoch_post_combination = None
        self.number_of_channels = None
        self.vector_size = None
        self.final_vector_size = None

    def setup(self, original_vector_size: int, initial_number_of_vectors_per_epoch: int,  number_of_channels: int,
              final_linear_projection_to_given_vector_size: Union[int, None] = None):
        assert not self.__setup_done_flag

        self.number_of_channels = number_of_channels

        self.vector_size = original_vector_size

        self.number_of_vectors_per_epoch_post_combination = self.combination_layer.setup(
            initial_number_of_vectors_per_epoch, self.vector_size, self.number_of_channels,
            self.INDEX_OF_CHANNEL_DIMENSION)

        if final_linear_projection_to_given_vector_size is None:
            self.final_vector_size = self.vector_size
        else:
            self.final_vector_size = final_linear_projection_to_given_vector_size
            self.final_linear_projection_layer = Linear(self.vector_size, self.final_vector_size)

        self.__setup_done_flag = True
        return self.final_vector_size, self.number_of_vectors_per_epoch_post_combination

    # sequence_of_vectorized_spd_matrices of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, vector_size)
    # output of shape (batch_size, sequences_of_epochs_length, number_of_matrices_per_epoch * number_of_channels, final_vector_size)
    def forward(self, sequence_of_vectorized_spd_matrices: torch.Tensor):
        assert self.__setup_done_flag

        assert sequence_of_vectorized_spd_matrices.shape[self.INDEX_OF_CHANNEL_DIMENSION] == self.number_of_channels
        assert sequence_of_vectorized_spd_matrices.shape[-1] == self.vector_size

        # (batch_size, sequences_of_epochs_length, number_of_matrices_per_epoch * number_of_channels, vector_size)
        sequence_of_vectorized_matrices_with_channels_reorganized_in_succession = self.combination_layer(
            sequence_of_vectorized_spd_matrices)
        assert len(sequence_of_vectorized_matrices_with_channels_reorganized_in_succession.shape)\
               == len(sequence_of_vectorized_spd_matrices.shape) - 1
        assert sequence_of_vectorized_matrices_with_channels_reorganized_in_succession.shape[-1] == self.vector_size
        assert sequence_of_vectorized_matrices_with_channels_reorganized_in_succession.shape[-2]\
               == self.number_of_vectors_per_epoch_post_combination

        output_tensor = sequence_of_vectorized_matrices_with_channels_reorganized_in_succession
        if self.final_linear_projection_layer is not None:
            output_tensor = self.final_linear_projection_layer(output_tensor)
            assert output_tensor.shape[-1] == self.final_vector_size

        return output_tensor



