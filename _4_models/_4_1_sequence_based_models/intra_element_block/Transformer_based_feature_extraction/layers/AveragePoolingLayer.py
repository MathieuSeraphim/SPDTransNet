import torch
from torch.nn import Module, AdaptiveAvgPool2d


class AveragePoolingLayer(Module):

    def __init__(self):
        super(AveragePoolingLayer, self).__init__()
        self.__setup_done_flag = False

        self.pooling_layer = None
        self.input_sequence_length = None
        self.output_sequence_length = None
        self.vector_length = None
        self.squeeze_output_flag = None

    def setup(self, input_sequence_length: int, output_sequence_length: int, vector_length: int,
              remove_sequence_channel_if_of_size_one: bool = True):
        assert not self.__setup_done_flag

        assert 0 < output_sequence_length < input_sequence_length
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.vector_length = vector_length
        self.squeeze_output_flag = remove_sequence_channel_if_of_size_one and (self.output_sequence_length == 1)

        self.pooling_layer = AdaptiveAvgPool2d((self.output_sequence_length, self.vector_length))

        self.__setup_done_flag = True

    # x of shape (batch_size, input_sequence_length, vector_length)
    # output of shape (batch_size, output_sequence_length, vector_length) OR (batch_size, vector_length)
    def forward(self, x: torch.Tensor):
        assert self.__setup_done_flag

        batch_size, sequence_length, vector_length = x.shape
        assert (sequence_length, vector_length) == (self.input_sequence_length, self.vector_length)

        # (batch_size, output_sequence_length, vector_length)
        x = self.pooling_layer(x)

        assert len(x.shape) == 3
        assert x.shape == (batch_size, self.output_sequence_length, self.vector_length)

        if not self.squeeze_output_flag:
            return x

        x = x.squeeze(dim=1)
        assert len(x.shape) == 2
        assert x.shape == (batch_size, self.vector_length)

        return x

