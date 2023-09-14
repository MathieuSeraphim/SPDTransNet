from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.non_spd_blocks.TransformerBasedSequenceToSmallerFeaturesSequenceIntraElementBlock import \
    TransformerBasedSequenceToSmallerFeaturesSequenceIntraElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.SPDTransformerEncoderLayer import \
    SPDTransformerEncoderLayer


class TransformerBasedSPDSequenceToSmallerFeaturesSequenceIntraElementBlock(TransformerBasedSequenceToSmallerFeaturesSequenceIntraElementBlock):

    def __init__(self):
        super(TransformerBasedSPDSequenceToSmallerFeaturesSequenceIntraElementBlock, self).__init__()
        self.transformer_encoder = SPDTransformerEncoderLayer()
