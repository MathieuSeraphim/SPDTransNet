from typing import List
import torch
from torch.nn import Linear
from _4_models._4_1_sequence_based_models.classification_block.BaseClassificationBlock import BaseClassificationBlock
from _4_models._4_1_sequence_based_models.classification_block.classification_from_sequence_central_feature.layers.FullyConnectedLayer import \
    FullyConnectedLayer


class CentralGroupOfFeaturesInSequenceClassificationBlock(BaseClassificationBlock):

    def __init__(self):
        super(CentralGroupOfFeaturesInSequenceClassificationBlock, self).__init__()
        self.__setup_done_flag = False

        self.first_fc_layer = FullyConnectedLayer()
        self.second_fc_layer = FullyConnectedLayer()
        self.classification_layer = None

        self.sequence_length = None
        self.vector_size = None
        self.number_of_classes = None
        self.relevant_feature_indices = None
        self.number_of_relevant_features_to_use_for_classification = None

    def setup(self, sequence_length: int, vector_size: int, number_of_classes: int, relevant_feature_indices: List[int],
              fully_connected_intermediary_dimension: int, fully_connected_dropout_rate: float):
        assert not self.__setup_done_flag

        self.sequence_length = sequence_length
        self.vector_size = vector_size
        self.number_of_classes = number_of_classes

        self.relevant_feature_indices = sorted(relevant_feature_indices)
        self.number_of_relevant_features_to_use_for_classification = len(self.relevant_feature_indices)
        assert len(set(self.relevant_feature_indices)) == self.number_of_relevant_features_to_use_for_classification
        assert 0 < len(self.relevant_feature_indices)
        assert 0 <= relevant_feature_indices[0] <= relevant_feature_indices[-1] < self.sequence_length

        final_features_size = self.vector_size * self.number_of_relevant_features_to_use_for_classification

        self.first_fc_layer.setup(final_features_size, fully_connected_intermediary_dimension,
                                  fully_connected_dropout_rate)
        self.second_fc_layer.setup(fully_connected_intermediary_dimension, fully_connected_intermediary_dimension,
                                   fully_connected_dropout_rate)
        self.classification_layer = Linear(fully_connected_intermediary_dimension, number_of_classes)

        self.__setup_done_flag = True

    # sequence_of_element_descriptor_vectors of shape (batch_size, sequence_length, vector_size)
    # output of shape (batch_size, number_of_classes)
    def forward(self, sequence_of_element_descriptor_vectors: torch.Tensor):
        assert self.__setup_done_flag

        assert len(sequence_of_element_descriptor_vectors.shape) == 3
        batch_size, sequence_length, vector_size = sequence_of_element_descriptor_vectors.shape
        assert (sequence_length, vector_size) == (self.sequence_length, self.vector_size)

        # (batch_size, number_of_relevant_features, vector_size)
        chosen_element_descriptor_vectors = sequence_of_element_descriptor_vectors[:, self.relevant_feature_indices, :]
        assert chosen_element_descriptor_vectors.shape == (batch_size,
                                                          self.number_of_relevant_features_to_use_for_classification,
                                                          self.vector_size)

        # (batch_size, number_of_relevant_features * vector_size)
        chosen_element_descriptor_vectors = chosen_element_descriptor_vectors.view(batch_size, -1)

        x = self.first_fc_layer(chosen_element_descriptor_vectors)
        x = self.second_fc_layer(x)
        logits = self.classification_layer(x)

        assert logits.shape == (batch_size, self.number_of_classes)

        # (batch_size, number_of_classes)
        return logits






