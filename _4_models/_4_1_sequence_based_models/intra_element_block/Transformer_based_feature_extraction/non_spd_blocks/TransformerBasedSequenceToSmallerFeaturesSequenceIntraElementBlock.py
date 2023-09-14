import torch
from _4_models._4_1_sequence_based_models.intra_element_block.BaseIntraElementBlock import BaseIntraElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.AveragePoolingLayer import \
    AveragePoolingLayer
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.LearnableSinusoidalPositionalEncodingLayer import \
    LearnableSinusoidalPositionalEncodingLayer
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.dependencies.StandardTransformerEncoderLayer import \
    StandardTransformerEncoderLayer


class TransformerBasedSequenceToSmallerFeaturesSequenceIntraElementBlock(BaseIntraElementBlock):

    def __init__(self):
        super(TransformerBasedSequenceToSmallerFeaturesSequenceIntraElementBlock, self).__init__()
        self.__setup_done_flag = False

        self.positional_encoding = LearnableSinusoidalPositionalEncodingLayer()
        self.transformer_encoder = StandardTransformerEncoderLayer()
        self.average_pooling = AveragePoolingLayer()

        self.sequence_length = None
        self.output_sequence_length = None
        self.number_of_vectors_per_element = None
        self.number_of_output_features = None
        self.vector_size = None

    def setup(self, sequence_length: int, number_of_vectors_per_element: int, number_of_output_features: int,
              vector_size: int, number_of_encoder_heads: int, encoder_feedforward_dimension: int,
              encoder_dropout_rate: float, number_of_encoder_layers: int):
        assert not self.__setup_done_flag

        assert 0 < sequence_length
        assert 0 < vector_size
        assert 0 < number_of_output_features < number_of_vectors_per_element

        self.sequence_length = sequence_length
        self.number_of_vectors_per_element = number_of_vectors_per_element
        self.number_of_output_features = number_of_output_features
        self.output_sequence_length = self.sequence_length * self.number_of_output_features
        self.vector_size = vector_size

        self.positional_encoding.setup(number_of_vectors_per_element, vector_size)
        self.transformer_encoder.setup(d_model=self.vector_size, nhead=number_of_encoder_heads,
                                       dim_feedforward=encoder_feedforward_dimension, dropout=encoder_dropout_rate,
                                       num_layers=number_of_encoder_layers)
        self.average_pooling.setup(self.number_of_vectors_per_element, self.number_of_output_features, self.vector_size, False)

        self.__setup_done_flag = True
        return self.output_sequence_length

    # element_as_sequence_of_vectors of shape (batch_size, sequence_length, number_of_vectors_per_element, vector_size)
    # output of shape (batch_size, sequence_length, number_of_output_features, vector_size)
    def forward(self, element_as_sequence_of_vectors: torch.Tensor):
        assert self.__setup_done_flag

        assert len(element_as_sequence_of_vectors.shape) == 4
        batch_size, sequence_length, number_of_vectors_per_element, vector_size\
            = element_as_sequence_of_vectors.shape
        assert (sequence_length, number_of_vectors_per_element, vector_size)\
               == (self.sequence_length, self.number_of_vectors_per_element, self.vector_size)

        # (batch_size * sequence_length, number_of_vectors_per_element, vector_size)
        element_as_sequence_of_vectors = element_as_sequence_of_vectors.view(-1, number_of_vectors_per_element, vector_size)

        encoded_sequence_of_vectors = self.positional_encoding(element_as_sequence_of_vectors)
        transformed_sequence_of_vectors = self.transformer_encoder(encoded_sequence_of_vectors)

        # (batch_size * sequence_length, number_of_output_features, vector_size)
        averaged_transformed_vectors = self.average_pooling(transformed_sequence_of_vectors)

        # (batch_size, sequence_length * number_of_output_features, vector_size)
        output_element_feature_vectors = averaged_transformed_vectors.view(batch_size, self.output_sequence_length,
                                                                           vector_size)

        return output_element_feature_vectors


