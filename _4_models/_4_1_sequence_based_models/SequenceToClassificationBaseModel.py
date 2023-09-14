import warnings
from typing import List, Dict, Any, Union
from _4_models.BaseModel import BaseModel
from _4_models._4_1_sequence_based_models.classification_block.BaseClassificationBlock import BaseClassificationBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.BaseDataFormattingBlock import BaseDataFormattingBlock
from _4_models._4_1_sequence_based_models.inter_element_block.BaseInterElementBlock import BaseInterElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.BaseIntraElementBlock import BaseIntraElementBlock
from _4_models.utils import check_block_import


class SequenceToClassificationBaseModel(BaseModel):

    def __init__(self, loss_function_config_dict: Dict[str, Any], class_labels_list: List[str],
                 data_formatting_block: BaseDataFormattingBlock, intra_element_block: BaseIntraElementBlock,
                 inter_element_block: BaseInterElementBlock, classification_block: BaseClassificationBlock,
                 learning_rate: float, optimisation_config_dict: Union[Dict[str, Any], None] = None):
        super(SequenceToClassificationBaseModel, self).__init__(loss_function_config_dict, class_labels_list,
                                                                learning_rate, optimisation_config_dict)
        self.__setup_done_flag = False

        self.data_formatting_block = check_block_import(data_formatting_block)
        self.intra_element_block = check_block_import(intra_element_block)
        self.inter_element_block = check_block_import(inter_element_block)
        self.classification_block = check_block_import(classification_block)

        self.data_formatting_setup_kwargs = None
        self.intra_element_setup_kwargs = None
        self.inter_element_setup_kwargs = None
        self.classification_setup_kwargs = None

    def setup(self, stage: str):
        if not self.__setup_done_flag:
            self.data_formatting_block.setup(**self.data_formatting_setup_kwargs)
            self.intra_element_block.setup(**self.intra_element_setup_kwargs)
            self.inter_element_block.setup(**self.inter_element_setup_kwargs)
            self.classification_block.setup(**self.classification_setup_kwargs)
            self.__setup_done_flag = True

    def preprocess_input(self, input, set_name):
        raise NotImplementedError

    def forward(self, **inputs):
        assert self.__setup_done_flag
        formatted_sequence = self.data_formatting_block(**inputs)
        feature_sequence = self.intra_element_block(formatted_sequence)
        final_features = self.inter_element_block(feature_sequence)
        classification_logits = self.classification_block(final_features)
        return classification_logits

    def obtain_example_input_array(self):
        if not type(self) == SequenceToClassificationBaseModel:
            raise NotImplementedError
        warnings.warn("This is a base instance of a sequence-based model, that must be inherited.")
