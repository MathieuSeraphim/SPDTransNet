from typing import Union, Callable, Optional
from torch import Tensor
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.StructurePreservingMultiheadAttention import \
    StructurePreservingMultiheadAttention


class SPDTransformerEncoderSublayer(TransformerEncoderLayer):

    def __init__(self,  d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = True, norm_first: bool = False, device=None, dtype=None):
        assert batch_first

        super(SPDTransformerEncoderSublayer, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype
        )

        # We define a multihead attention where V is fed in unchanged
        self.self_attn = StructurePreservingMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, device=device,
                                                               dtype=dtype)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None):
        assert (src_mask, src_key_padding_mask) == (None, None)  # Other inputs unsupported
        output = super(SPDTransformerEncoderSublayer, self).forward(src, src_mask, src_key_padding_mask)
        assert output.shape == src.shape
        return output



