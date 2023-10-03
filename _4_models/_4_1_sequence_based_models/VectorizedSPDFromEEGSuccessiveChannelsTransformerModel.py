from typing import List, Union, Dict, Any
import torch
from _4_models._4_1_sequence_based_models.SequenceToClassificationBaseModel import SequenceToClassificationBaseModel
from _4_models._4_1_sequence_based_models.classification_block.BaseClassificationBlock import BaseClassificationBlock
from _4_models._4_1_sequence_based_models.classification_block.classification_from_sequence_central_feature.CentralGroupOfFeaturesInSequenceClassificationBlock import \
    CentralGroupOfFeaturesInSequenceClassificationBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.BaseDataFormattingBlock import BaseDataFormattingBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.VectorizedSPDFromEEGDataMultichannelSuccessionReformattingBlock import \
    VectorizedSPDFromEEGDataMultichannelSuccessionReformattingBlock
from _4_models._4_1_sequence_based_models.inter_element_block.BaseInterElementBlock import BaseInterElementBlock
from _4_models._4_1_sequence_based_models.inter_element_block.Transformer_based_feature_comparison.TransformerBasedSPDLearnablePositionalEncodingSequenceToSequenceInterElementBlock import \
    TransformerBasedSPDLearnablePositionalEncodingSequenceToSequenceInterElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.BaseIntraElementBlock import BaseIntraElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.TransformerBasedSPDSequenceToSmallerFeaturesSequenceIntraElementBlock import \
    TransformerBasedSPDSequenceToSmallerFeaturesSequenceIntraElementBlock


# This class corresponds to the SPDTransNet model in the paper.
class VectorizedSPDFromEEGSuccessiveChannelsTransformerModel(SequenceToClassificationBaseModel):

    COMPATIBLE_DATA_FORMATTING_BLOCKS = (VectorizedSPDFromEEGDataMultichannelSuccessionReformattingBlock,)
    COMPATIBLE_INTRA_ELEMENT_BLOCKS = (TransformerBasedSPDSequenceToSmallerFeaturesSequenceIntraElementBlock,)
    COMPATIBLE_INTER_ELEMENT_BLOCKS = (TransformerBasedSPDLearnablePositionalEncodingSequenceToSequenceInterElementBlock,)
    COMPATIBLE_CLASSIFICATION_BLOCKS = (CentralGroupOfFeaturesInSequenceClassificationBlock,)

    def __init__(
            self, loss_function_config_dict: Dict[str, Any], class_labels_list: List[str],
            data_formatting_block: BaseDataFormattingBlock, intra_element_block: BaseIntraElementBlock,
            inter_element_block: BaseInterElementBlock, classification_block: BaseClassificationBlock,
            number_of_eeg_signals: int, number_of_channels: int, extra_epochs_on_each_side: int,
            number_of_subdivisions_per_epoch: int, matrix_augmentation_size: int,
            final_linear_projection_to_given_vector_size: Union[int, None], number_of_intra_epoch_encoder_heads: int,
            intra_epoch_encoder_feedforward_dimension: int, intra_epoch_encoder_dropout_rate: float,
            number_of_intra_epoch_encoder_layers: int, number_of_inter_epoch_encoder_heads: int,
            inter_epoch_encoder_feedforward_dimension: int, inter_epoch_encoder_dropout_rate: float,
            number_of_inter_epoch_encoder_layers: int, fully_connected_intermediary_dimension: int,
            fully_connected_dropout_rate: float, learning_rate: float,
            number_of_epoch_wise_feature_vectors: Union[int, None] = None,
            optimisation_config_dict: Union[Dict[str, Any], None] = None
    ):
        r"""
        Constructor for the class, including arguments to initialize its component blocks.
        :param loss_function_config_dict: configuration of the loss function strategy, with a limited number of
         pre-defined options. See the relevant code for more details.
        :param class_labels_list: the label names, as defined in the Preprocessor. Order matters!
        :param data_formatting_block: an object of the class utilized for data formatting - i.e. everything that
         precedes the first analysis block. Parameters will be passed to it during setup.
        :param intra_element_block: an object of the class utilized for intra-element analysis - for us, the intra-epoch
         Transformer encoder followed by an average pooling. Parameters will be passed to it during setup.
        :param inter_element_block: an object of the class utilized for inter-element analysis - for us, the inter-epoch
         Transformer encoder.  Parameters will be passed to it during setup.
        :param classification_block: an object of the class utilized for the final classification - for us, a couple of
         fully connected layers followed by a final projection onto R^5. Parameters will be passed to it during setup.
        :param number_of_eeg_signals: also the size of the pre-augmentation covariance matrices - 8 for us.
        :param number_of_channels: self-explanatory.
        :param extra_epochs_on_each_side: number of context epochs surrounding the central epoch to be classified.
         Total input sequence length is 2 * extra_epochs_on_each_side + 1.
        :param number_of_subdivisions_per_epoch: the length of the intra-epoch timeseries - 30 for us.
        :param matrix_augmentation_size: the matrices used for augmentation are of shape
         (number_of_eeg_signals * matrix_augmentation_size). The final matrices are in
         SPD(number_of_eeg_signals + matrix_augmentation_size).
        :param final_linear_projection_to_given_vector_size: after tokenization, the SPD-derived tokens are linearly
         mapped into vectors of length final_linear_projection_to_given_vector_size. For us, must be a triangular
         number.
        :param number_of_intra_epoch_encoder_heads: for the Transformer encoder within the intra_element_block object.
        :param intra_epoch_encoder_feedforward_dimension: for the Transformer encoder within the intra_element_block
         object. For us, must be a triangular number.
        :param intra_epoch_encoder_dropout_rate: for the Transformer encoder within the intra_element_block object.
        :param number_of_intra_epoch_encoder_layers: for the Transformer encoder within the intra_element_block object.
         N_{intra} on the architecture's figure.
        :param number_of_inter_epoch_encoder_heads: for the Transformer encoder within the inter_element_block object.
        :param inter_epoch_encoder_feedforward_dimension: for the Transformer encoder within the inter_element_block
         object. For us, must be a triangular number.
        :param inter_epoch_encoder_dropout_rate: for the Transformer encoder within the inter_element_block object.
        :param number_of_inter_epoch_encoder_layers: for the Transformer encoder within the inter_element_block object.
         N_{inter} on the architecture's figure.
        :param fully_connected_intermediary_dimension: for the FC layers in the classification_block object. For us,
         must be a triangular number.
        :param fully_connected_dropout_rate: for the FC layers in the classification_block object.
        :param learning_rate: self-explanatory.
        :param number_of_epoch_wise_feature_vectors: number of feature vectors output by the intra-epoch step, per
         epoch. t in the paper. Must divide number_of_subdivisions_per_epoch.
        :param optimisation_config_dict: configuration of the optimization strategy, with a limited number of
         pre-defined options. See the relevant code for more details.
        """
        self.save_hyperparameters(logger=False,
                                  ignore=["data_formatting_block", "intra_element_block", "inter_element_block",
                                          "classification_block"])

        augmented_matrix_size = number_of_eeg_signals + matrix_augmentation_size
        self.original_vector_size = (augmented_matrix_size * (augmented_matrix_size + 1)) / 2
        assert self.original_vector_size == int(self.original_vector_size)
        self.original_vector_size = int(self.original_vector_size)
        self.original_number_of_channels = number_of_channels
        self.central_epoch_index = extra_epochs_on_each_side
        self.sequences_of_epochs_length = 2*extra_epochs_on_each_side + 1
        self.number_of_subdivisions_per_epoch = number_of_subdivisions_per_epoch

        super(VectorizedSPDFromEEGSuccessiveChannelsTransformerModel, self).__init__(loss_function_config_dict,
                                                                                     class_labels_list,
                                                                                     data_formatting_block,
                                                                                     intra_element_block,
                                                                                     inter_element_block,
                                                                                     classification_block,
                                                                                     learning_rate,
                                                                                     optimisation_config_dict)

        self.loss_strategies = ["central_epoch", "central_epochs_transition_penalty", "cross_entropy_with_label_smoothing"]
        if loss_function_config_dict["name"] in ["cross_entropy", "cross_entropy_with_label_smoothing"]:
            self.loss_strategy = "central_epoch"
        elif loss_function_config_dict["name"] == "cross_entropy_with_transition_penalty":
            assert loss_function_config_dict["args"]["number_of_classes"] == self.number_of_classes
            assert self.sequences_of_epochs_length > 1
            self.loss_strategy = "central_epochs_transition_penalty"
        else:
            raise NotImplementedError

        assert isinstance(self.data_formatting_block, self.COMPATIBLE_DATA_FORMATTING_BLOCKS)
        assert isinstance(self.intra_element_block, self.COMPATIBLE_INTRA_ELEMENT_BLOCKS)
        assert isinstance(self.inter_element_block, self.COMPATIBLE_INTER_ELEMENT_BLOCKS)
        assert isinstance(self.classification_block, self.COMPATIBLE_CLASSIFICATION_BLOCKS)

        if number_of_epoch_wise_feature_vectors is None:
            number_of_epoch_wise_feature_vectors = self.original_number_of_channels
        else:
            number_of_vectors_pooled_into_feature_vector = (self.number_of_subdivisions_per_epoch * self.original_number_of_channels) / number_of_epoch_wise_feature_vectors
            assert int(number_of_vectors_pooled_into_feature_vector) == number_of_vectors_pooled_into_feature_vector

        extended_central_epoch_range_start = self.central_epoch_index * number_of_epoch_wise_feature_vectors
        extended_central_epoch_range_end = extended_central_epoch_range_start + number_of_epoch_wise_feature_vectors
        extended_central_epoch_indices = range(extended_central_epoch_range_start, extended_central_epoch_range_end)

        self.data_formatting_setup_kwargs = {
            "original_vector_size": self.original_vector_size,
            "initial_number_of_vectors_per_epoch": self.number_of_subdivisions_per_epoch,
            "number_of_channels": self.original_number_of_channels,
            "final_linear_projection_to_given_vector_size": final_linear_projection_to_given_vector_size
        }

        self.intra_element_setup_kwargs = {
            "sequence_length": self.sequences_of_epochs_length,
            "number_of_output_features": number_of_epoch_wise_feature_vectors,
            "number_of_encoder_heads": number_of_intra_epoch_encoder_heads,
            "encoder_feedforward_dimension": intra_epoch_encoder_feedforward_dimension,
            "encoder_dropout_rate": intra_epoch_encoder_dropout_rate,
            "number_of_encoder_layers": number_of_intra_epoch_encoder_layers
        }

        self.inter_element_setup_kwargs = {
            "number_of_encoder_heads": number_of_inter_epoch_encoder_heads,
            "encoder_feedforward_dimension": inter_epoch_encoder_feedforward_dimension,
            "encoder_dropout_rate": inter_epoch_encoder_dropout_rate,
            "number_of_encoder_layers": number_of_inter_epoch_encoder_layers
        }

        self.classification_setup_kwargs = {
            "number_of_classes": self.number_of_classes,
            "relevant_feature_indices": extended_central_epoch_indices,
            "fully_connected_intermediary_dimension": fully_connected_intermediary_dimension,
            "fully_connected_dropout_rate": fully_connected_dropout_rate
        }

    def send_hparams_to_logger(self):
        hparams_dict = dict(self.hparams)

        hparams_dict["data_formatting_block"] = self.get_block_dict(self.data_formatting_block)
        hparams_dict["intra_element_block"] = self.get_block_dict(self.intra_element_block)
        hparams_dict["inter_element_block"] = self.get_block_dict(self.inter_element_block)
        hparams_dict["classification_block"] = self.get_block_dict(self.classification_block)

        model_dict = self.get_block_dict(self)
        model_dict["init_args"] = hparams_dict

        output_dict = {"model": model_dict}
        self.hparams.clear()
        self.hparams.update(output_dict)
        self.save_hyperparameters()

    def setup(self, stage: str):
        if not self._SequenceToClassificationBaseModel__setup_done_flag:
            self.send_hparams_to_logger()

            vector_size, extended_sequences_of_epochs_length = self.data_formatting_block.setup(
                **self.data_formatting_setup_kwargs)

            extended_sequence_length = self.intra_element_block.setup(
                vector_size=vector_size, number_of_vectors_per_element=extended_sequences_of_epochs_length,
                **self.intra_element_setup_kwargs)

            self.inter_element_block.setup(sequence_length=extended_sequence_length, vector_size=vector_size,
                                           **self.inter_element_setup_kwargs)

            self.classification_block.setup(sequence_length=extended_sequence_length, vector_size=vector_size,
                                            **self.classification_setup_kwargs)

            self._SequenceToClassificationBaseModel__setup_done_flag = True

    # sequence_of_vectorized_spd_matrices of shape (batch_size, number_of_channels, sequences_of_epochs_length, number_of_matrices_per_epoch, vector_size)
    # output of shape (batch_size, number_of_classes)
    def forward(self, sequence_of_vectorized_spd_matrices: torch.Tensor):

        return super(VectorizedSPDFromEEGSuccessiveChannelsTransformerModel, self).forward(
            sequence_of_vectorized_spd_matrices=sequence_of_vectorized_spd_matrices
        )

    def preprocess_input(self, input, set_name):
        model_input_dict = {
            "sequence_of_vectorized_spd_matrices": None
        }

        sequence_of_vectorized_spd_matrices = input["vectorized matrices"]
        assert sequence_of_vectorized_spd_matrices.shape[1:] == (self.original_number_of_channels, self.sequences_of_epochs_length, self.number_of_subdivisions_per_epoch, self.original_vector_size)
        model_input_dict["sequence_of_vectorized_spd_matrices"] = sequence_of_vectorized_spd_matrices

        return model_input_dict

    def process_loss_function_inputs(self, predictions, groundtruth, set_name):
        assert self.loss_strategy in self.loss_strategies

        assert len(predictions.shape) == 2 and len(groundtruth.shape) == 2
        assert predictions.shape[0] == groundtruth.shape[0]
        assert groundtruth.shape[1] == self.sequences_of_epochs_length
        assert predictions.shape[1] == self.number_of_classes

        if self.loss_strategy == "central_epoch":
            return {"input": predictions, "target": groundtruth[:, self.central_epoch_index].long()}
        if self.loss_strategy == "central_epochs_transition_penalty":
            return {"input": predictions,
                    "target": groundtruth[:, self.central_epoch_index].long(),
                    "previous_epoch_target": groundtruth[:, self.central_epoch_index - 1].long()}

    def obtain_example_input_array(self):
        placeholder_batch_size = 1
        sequence_of_vectorized_spd_matrices = torch.rand(placeholder_batch_size, self.original_number_of_channels, self.sequences_of_epochs_length, self.number_of_subdivisions_per_epoch, self.original_vector_size)
        self.example_input_array = (sequence_of_vectorized_spd_matrices,)


