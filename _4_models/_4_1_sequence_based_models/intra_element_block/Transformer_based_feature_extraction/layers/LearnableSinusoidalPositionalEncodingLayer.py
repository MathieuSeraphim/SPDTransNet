from torch.nn import Module, Linear, GELU
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.dependencies.PositionalEncodingLayer import \
    PositionalEncodingLayer


# As described in https://aclanthology.org/2022.findings-aacl.42
class LearnableSinusoidalPositionalEncodingLayer(Module):

    def __init__(self):
        super(LearnableSinusoidalPositionalEncodingLayer, self).__init__()
        self.__setup_done_flag = False

        self.sinusoidal_positional_encoding_layer = PositionalEncodingLayer()

        self.first_linear_layer = None
        self.activation_function_layer = GELU()
        self.second_linear_layer = None

    def setup(self, sequence_length: int, vector_length: int):
        assert not self.__setup_done_flag

        self.sinusoidal_positional_encoding_layer.setup(sequence_length, vector_length)
        self.first_linear_layer = Linear(vector_length, vector_length, bias=True)
        self.second_linear_layer = Linear(vector_length, vector_length, bias=True)

        self.__setup_done_flag = True

    # x of shape (batch_size, sequence_length, vector_length)
    def forward(self, x):
        assert self.__setup_done_flag

        x = self.sinusoidal_positional_encoding_layer(x)
        x = self.first_linear_layer(x)
        x = self.activation_function_layer(x)
        x = self.second_linear_layer(x)

        return x

