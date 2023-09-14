import math
from typing import Optional
import torch
from torch import Tensor
from torch.nn import MultiheadAttention, Linear, Parameter
from torch.nn.functional import linear, softmax, dropout


class StructurePreservingMultiheadAttention(MultiheadAttention):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=True, device=None, dtype=None):
        assert (add_bias_kv, add_zero_attn, kdim, vdim) == (False, False, None, None)  # Other inputs not supported
        assert batch_first

        super(StructurePreservingMultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                                                                    kdim, vdim, batch_first, device, dtype)

        linear_transformations_weights = self.in_proj_weight
        linear_transformations_bias = self.in_proj_bias

        # As is done in torch.nn.functional._in_projection_packed
        query_weights, key_weights, _ = linear_transformations_weights.chunk(3)
        self.query_weights = Parameter(query_weights.detach())
        self.key_weights = Parameter(key_weights.detach())
        if bias:
            query_bias, key_bias, _ = linear_transformations_bias.chunk(3)
            self.query_bias = Parameter(query_bias.detach())
            self.key_bias = Parameter(key_bias.detach())
        else:
            self.query_bias, self.key_bias = None, None

        assert self.query_weights.shape == self.key_weights.shape == (self.embed_dim, self.embed_dim)

        if bias:
            assert self.query_bias.requires_grad == self.key_bias.requires_grad == True
            assert self.query_bias.shape == self.key_bias.shape == (self.embed_dim,)

        self.weighted_sum_of_head_wise_attention_tensors = Linear(in_features=num_heads, out_features=1, bias=False)

    # query, key, value of shape (batch_size, sequence_length, vector_size)
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = False, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = False):
        assert (key_padding_mask, need_weights, attn_mask, average_attn_weights) == (None, False, None, False)

        assert len(value.shape) == 3
        assert query.shape == key.shape == value.shape
        batch_size, sequence_length, vector_size = value.shape
        assert vector_size == self.embed_dim

        # (sequence_length, batch_size, vector_size)
        if query is key:
            query = query.transpose(0, 1)
            key = query
        else:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)

        query_projected = linear(query, self.query_weights, self.query_bias)
        key_projected = linear(key, self.key_weights, self.key_bias)

        assert query_projected.shape == key_projected.shape == query.shape == key.shape

        # (sequence_length, batch_size * num_heads, head_dim)
        query_reshaped = query_projected.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)
        key_reshaped = key_projected.contiguous().view(sequence_length, batch_size * self.num_heads, self.head_dim)

        # (batch_size * num_heads, sequence_length, head_dim)
        query_reshaped = query_reshaped.transpose(0, 1)
        key_reshaped = key_reshaped.transpose(0, 1)

        dropout_p = self.dropout
        if not self.training:
            dropout_p = 0.0

        # (batch_size * num_heads, sequence_length, sequence_length)
        scaled_dot_product_attention_tensor = self.scaled_dot_product_attention_up_to_the_last_softmax(
            query_reshaped, key_reshaped, dropout_p)

        assert scaled_dot_product_attention_tensor.shape == (
            batch_size * self.num_heads, sequence_length, sequence_length)

        # (batch_size, sequence_length, sequence_length, num_heads)
        scaled_dot_product_attention_tensor_reshaped = scaled_dot_product_attention_tensor.view(
            batch_size, self.num_heads, sequence_length, sequence_length)
        scaled_dot_product_attention_tensor_reshaped = scaled_dot_product_attention_tensor_reshaped.permute(0, 2, 3, 1)

        # Linearly combine the attention matrices into one ber batch
        # (batch_size, sequence_length, sequence_length)
        scaled_dot_product_attention_tensor_combined = \
            self.weighted_sum_of_head_wise_attention_tensors(scaled_dot_product_attention_tensor_reshaped).squeeze(-1)

        assert scaled_dot_product_attention_tensor_combined.shape == (batch_size, sequence_length, sequence_length)

        output = torch.bmm(scaled_dot_product_attention_tensor_combined, value)
        assert output.shape == value.shape

        # (batch_size, sequence_length, vector_size)
        return output, None  # We impose need_weights as False

    # query, key of shape (N, sequence_length, head_dim)
    # output of shape (N, sequence_length, sequence_length)
    def scaled_dot_product_attention_up_to_the_last_softmax(self, query: Tensor, key: Tensor, dropout_p: float = 0.0):
        assert len(query.shape) == 3
        assert query.shape == key.shape
        batch_size, sequence_length, head_dim = query.shape
        assert head_dim == self.head_dim

        query = query / math.sqrt(head_dim)

        # (N, sequence_length, head_dim) * (N, head_dim, sequence_length) -> (N, sequence_length, sequence_length)
        attention_tensor = torch.bmm(query, key.transpose(-2, -1))

        attention_tensor = softmax(attention_tensor, dim=-1)
        if dropout_p > 0.0:
            attention_tensor = dropout(attention_tensor, p=dropout_p)

        # (N, sequence_length, sequence_length)
        return attention_tensor
