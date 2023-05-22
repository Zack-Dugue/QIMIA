import torch.nn as nn
import torch.nn.functional as F
import torch as th
from typing import Optional, Tuple

#stolen from Torch Text
class ScaledDotProduct(th.nn.Module):
    def __init__(self, dropout=0.0, batch_first=False) -> None:
        r"""Processes a projected query and key-value pair to apply
        scaled dot product attention.

        Args:
            dropout (float): probability of dropping an attention weight.
            batch_first: If ``True``, then the input and output tensors are provided
                as `(batch, seq, feature)`. Default: ``False``

        Examples::
            # >>> import torch as th, torchtext as tt
            # >>> SDP = tt.nn.ScaledDotProduct(dropout=0.1)
            # >>> q = th.randn(21, 256, 3)
            # >>> k = v = th.randn(21, 256, 3)
            # >>> attn_output, attn_weights = SDP(q, k, v)
            # >>> print(attn_output.shape, attn_weights.shape)
            th.Size([21, 256, 3]) th.Size([256, 21, 21])
        """
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(
        self,
        query: th.Tensor,
        key: th.Tensor,
        value: th.Tensor,
        attn_mask: Optional[th.Tensor] = None,
        bias_k: Optional[th.Tensor] = None,
        bias_v: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        r"""Uses a scaled dot product with the projected key-value pair to update
        the projected query.

        Args:
            query (Tensor): Projected query
            key (Tensor): Projected key
            value (Tensor): Projected value
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            bias_k (Tensor, optional): one more key and value sequence to be added to keys at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide
                ``bias_v``.
            bias_v (Tensor, optional): one more key and value sequence to be added to values at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should also provide
                ``bias_k``.

        Shape:
            - query: :math:`(..., L, N * H, E / H)`
            - key: :math:`(..., S, N * H, E / H)`
            - value: :math:`(..., S, N * H, E / H)`
            - attn_mask: :math:`(N * H, L, S)`, positions with ``True`` are not allowed to attend
                while ``False`` values will be unchanged.
            - bias_k and bias_v:bias: :math:`(1, N * H, E / H)`

            - Output: :math:`(..., L, N * H, E / H)`, :math:`(N * H, L, S)`

            Note: It's optional to have the query/key/value inputs with more than three dimensions (for broadcast purpose).
                The ScaledDotProduct module will operate on the last three dimensions.

            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        if self.batch_first:
            query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)

        if bias_k is not None and bias_v is not None:
            assert (
                key.size(-1) == bias_k.size(-1) and key.size(-2) == bias_k.size(-2) and bias_k.size(-3) == 1
            ), "Shape of bias_k is not supported"
            assert (
                value.size(-1) == bias_v.size(-1) and value.size(-2) == bias_v.size(-2) and bias_v.size(-3) == 1
            ), "Shape of bias_v is not supported"
            key = th.cat([key, bias_k])
            value = th.cat([value, bias_v])
            if attn_mask is not None:
                attn_mask = th.nn.functional.pad(attn_mask, (0, 1))

        tgt_len, head_dim = query.size(-3), query.size(-1)
        # assert query.size(-1) == key.size(-1) == value.size(-1), "The feature dim of query, key, value must be equal."
        assert key.size(1) == value.size(1), "Shape of key, value must match"
        src_len = key.size(-3)
        batch_heads = max(query.size(-2), key.size(-2))

        # Scale query
        query, key, value = query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3)
        query = query * (float(head_dim) ** -0.5)
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise RuntimeError("attn_mask must be a 3D tensor.")
            if (
                (attn_mask.size(-1) != src_len)
                or (attn_mask.size(-2) != tgt_len)
                or (attn_mask.size(-3) != 1 and attn_mask.size(-3) != batch_heads)
            ):
                raise RuntimeError("The size of the attn_mask is not correct.")
            if attn_mask.dtype != th.bool:
                raise RuntimeError("Only bool tensor is supported for attn_mask")

        # Dot product of q, k
        attn_output_weights = th.matmul(query, key.transpose(-2, -1))
        if attn_mask is not None:
            attn_output_weights.masked_fill_(
                attn_mask,
                -1e8,
            )
        attn_output_weights = th.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = th.nn.functional.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = th.matmul(attn_output_weights, value)

        if self.batch_first:
            return attn_output, attn_output_weights
        else:
            return attn_output.transpose(-3, -2), attn_output_weights