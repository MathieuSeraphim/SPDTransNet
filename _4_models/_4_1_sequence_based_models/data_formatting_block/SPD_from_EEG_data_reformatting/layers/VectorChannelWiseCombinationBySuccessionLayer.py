import torch
from torch.nn import Module


class VectorChannelWiseCombinationBySuccessionLayer(Module):

    def __init__(self):
        super(VectorChannelWiseCombinationBySuccessionLayer, self).__init__()
        self.__setup_done_flag = False
        self.__just_remove_channel_flag = True

        self.sequence_length = None
        self.vector_size = None
        self.number_of_channels = None
        self.index_of_channel_dimension = None
        self.combined_sequence_length = None

        self.mask = None

    def setup(self, sequence_length: int, vector_size: int, number_of_channels: int, index_of_channel_dimension: int):
        assert not self.__setup_done_flag
        self.sequence_length = sequence_length
        self.vector_size = vector_size
        self.number_of_channels = number_of_channels
        self.index_of_channel_dimension = index_of_channel_dimension
        assert self.vector_size > 0
        assert number_of_channels >= 1
        self.combined_sequence_length = self.sequence_length * self.number_of_channels

        self.__just_remove_channel_flag = number_of_channels == 1
        self.__setup_done_flag = True
        return self.combined_sequence_length

    # spd_matrices_to_vectorize of shape (<prior dimensions>, channels, <later dimensions>, sequence_length, vector_size)
    # output of shape (<prior dimensions>, <later dimensions>, sequence_length * channels, vector_size)
    def forward(self, vectors_to_combine: torch.Tensor):
        assert self.__setup_done_flag

        pre_combination_shape = vectors_to_combine.shape
        assert len(pre_combination_shape) >= 3 and self.index_of_channel_dimension != len(pre_combination_shape) - 2
        assert pre_combination_shape[self.index_of_channel_dimension] == self.number_of_channels
        assert pre_combination_shape[-2] == self.sequence_length
        assert pre_combination_shape[-1] == self.vector_size

        if self.__just_remove_channel_flag:
            output_vector = vectors_to_combine.unsqueeze(dim=self.index_of_channel_dimension)
            assert len(output_vector.shape) == len(pre_combination_shape) - 1
            return output_vector

        vectors_in_channel_wise_list = torch.unbind(vectors_to_combine, dim=self.index_of_channel_dimension)
        output_vectors = torch.cat(vectors_in_channel_wise_list, dim=-2)

        assert output_vectors.shape == tuple([pre_combination_shape[i]
                                              for i in range(len(pre_combination_shape)-2)
                                              if i != self.index_of_channel_dimension]
                                             + [self.combined_sequence_length, self.vector_size])

        # (<prior dimensions>, <later dimensions>, sequence_length * channels, vector_size)
        return output_vectors











