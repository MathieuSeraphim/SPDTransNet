import math
import torch
from torch.nn import Module, Parameter


class PositionalEncodingLayer(Module):

    def __init__(self):
        super(PositionalEncodingLayer, self).__init__()
        self.__setup_done_flag = False

        self.positional_encoding = None

    def setup(self, sequence_length: int, vector_length: int):
        assert not self.__setup_done_flag

        positional_encoding_vector_length = vector_length
        positional_encoding = self.positionalencoding1d(d_model=positional_encoding_vector_length,
                                                        length=sequence_length, accepts_odd_numbers_mode=True)

        # positional_encoding of shape (sequence_length, vector_length)
        self.positional_encoding = Parameter(positional_encoding, requires_grad=False)

        self.__setup_done_flag = True

    # Taken from https://github.com/wzlxjtu/PositionalEncoding2D
    # And modified
    @staticmethod
    def positionalencoding1d(d_model, length, accepts_odd_numbers_mode=False):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        even_d_model = d_model
        if d_model % 2 != 0:
            if not accepts_odd_numbers_mode:
                raise ValueError("Cannot use sin/cos positional encoding with "
                                 "odd dim (got dim={:d})".format(d_model))
            else:
                even_d_model = d_model + 1

        pe = torch.zeros(length, even_d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, even_d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        if d_model % 2 != 0:
            pe = pe[:, :-1]

        return pe

    # x of shape (batch_size, sequence_length, vector_length)
    def forward(self, x):
        assert self.__setup_done_flag
        return x + self.positional_encoding

