import torch
from _4_models._4_1_sequence_based_models.inter_element_block.BaseInterElementBlock import BaseInterElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.LearnableSinusoidalPositionalEncodingLayer import \
    LearnableSinusoidalPositionalEncodingLayer
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.dependencies.StandardTransformerEncoderLayer import \
    StandardTransformerEncoderLayer


class TransformerBasedLearnablePositionalEncodingSequenceToSequenceInterElementBlock(BaseInterElementBlock):

    def __init__(self):
        super(TransformerBasedLearnablePositionalEncodingSequenceToSequenceInterElementBlock, self).__init__()
        self.__setup_done_flag = False

        self.positional_encoding = LearnableSinusoidalPositionalEncodingLayer()
        self.transformer_encoder = StandardTransformerEncoderLayer()

        self.sequence_length = None
        self. vector_size = None

    def setup(self, sequence_length: int, vector_size: int, number_of_encoder_heads: int,
              encoder_feedforward_dimension: int, encoder_dropout_rate: float, number_of_encoder_layers: int):
        assert not self.__setup_done_flag

        self.sequence_length = sequence_length
        self.vector_size = vector_size

        self.positional_encoding.setup(self.sequence_length, self.vector_size)
        self.transformer_encoder.setup(d_model=self.vector_size, nhead=number_of_encoder_heads,
                                       dim_feedforward=encoder_feedforward_dimension, dropout=encoder_dropout_rate,
                                       num_layers=number_of_encoder_layers)

        self.__setup_done_flag = True

    # sequence_of_element_descriptor_vectors of shape (batch_size, sequence_length, vector_size)
    # output of same shape
    def forward(self, sequence_of_element_descriptor_vectors: torch.Tensor):
        assert self.__setup_done_flag

        input_shape = sequence_of_element_descriptor_vectors.shape
        assert len(input_shape) == 3
        batch_size, sequence_length, vector_size = input_shape
        assert (sequence_length, vector_size) == (self.sequence_length, self.vector_size)

        encoded_sequence_of_vectors = self.positional_encoding(sequence_of_element_descriptor_vectors)
        transformed_sequence_of_vectors = self.transformer_encoder(encoded_sequence_of_vectors)

        assert transformed_sequence_of_vectors.shape == input_shape

        # (batch_size, sequence_length, vector_size)
        return transformed_sequence_of_vectors


