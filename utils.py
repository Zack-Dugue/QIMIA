import torch.nn as nn
import torch.nn.functional as F
import torch as th
from typing import Optional, Tuple
from functools import reduce
#Stolen from  https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
def factors(n):
    return reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))

def get_heads(key_dim,min_size=8):
    factor_list = factors(key_dim)
    factor_list.sort()
    for factor in factor_list:
        if factor >= min_size:
            return int(key_dim/factor)
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


#also stolen from torchtext:
class MultiheadAttentionContainer(th.nn.Module):
    def __init__(self, nhead, in_proj_container, attention_layer, out_proj, batch_first=False) -> None:
        r"""A multi-head attention container

        Args:
            nhead: the number of heads in the multiheadattention model
            in_proj_container: A container of multi-head in-projection linear layers (a.k.a nn.Linear).
            attention_layer: The custom attention layer. The input sent from MHA container to the attention layer
                is in the shape of `(..., L, N * H, E / H)` for query and `(..., S, N * H, E / H)` for key/value
                while the  output shape of the attention layer is expected to be `(..., L, N * H, E / H)`.
                The attention_layer needs to support broadcast if users want the overall MultiheadAttentionContainer
                with broadcast.
            out_proj: The multi-head out-projection layer (a.k.a nn.Linear).
            batch_first: If ``True``, then the input and output tensors are provided
                as `(..., N, L, E)`. Default: ``False``

        Examples::
            >>> import torch
            >>> from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
            >>> embed_dim, num_heads, bsz = 10, 5, 64
            >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim))
            >>> MHA = MultiheadAttentionContainer(num_heads,
                                                  in_proj_container,
                                                  ScaledDotProduct(),
                                                  torch.nn.Linear(embed_dim, embed_dim))
            >>> query = torch.rand((21, bsz, embed_dim))
            >>> key = value = torch.rand((16, bsz, embed_dim))
            >>> attn_output, attn_weights = MHA(query, key, value)
            >>> print(attn_output.shape)
            >>> torch.Size([21, 64, 10])
        """
        super(MultiheadAttentionContainer, self).__init__()
        self.nhead = nhead
        self.in_proj_container = in_proj_container
        self.attention_layer = attention_layer
        self.out_proj = out_proj
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
        r"""

        Args:
            query (Tensor): The query of the attention function.
                See "Attention Is All You Need" for more details.
            key (Tensor): The keys of the attention function.
                See "Attention Is All You Need" for more details.
            value (Tensor): The values of the attention function.
                See "Attention Is All You Need" for more details.
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            bias_k (Tensor, optional): one more key and value sequence to be added to keys at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide
                ``bias_v``.
            bias_v (Tensor, optional): one more key and value sequence to be added to values at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should also provide
                ``bias_k``.

        Shape:

            - Inputs:

                - query: :math:`(..., L, N, E)`
                - key: :math:`(..., S, N, E)`
                - value: :math:`(..., S, N, E)`
                - attn_mask, bias_k and bias_v: same with the shape of the corresponding args in attention layer.

            - Outputs:

                - attn_output: :math:`(..., L, N, E)`
                - attn_output_weights: :math:`(N * H, L, S)`

            Note: It's optional to have the query/key/value inputs with more than three dimensions (for broadcast purpose).
            The MultiheadAttentionContainer module will operate on the last three dimensions.

            where where L is the target length, S is the sequence length, H is the number of attention heads,
            N is the batch size, and E is the embedding dimension.
        """
        if self.batch_first:
            query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)

        tgt_len, src_len, bsz, embed_dim = query.size(-3), key.size(-3), query.size(-2), query.size(-1)
        q, k, v = self.in_proj_container(query, key, value)
        assert q.size(-1) % self.nhead == 0, "query's embed_dim must be divisible by the number of heads"
        head_dim = q.size(-1) // self.nhead
        q = q.reshape(tgt_len, bsz * self.nhead, head_dim)

        assert k.size(-1) % self.nhead == 0, "key's embed_dim must be divisible by the number of heads"
        head_dim = k.size(-1) // self.nhead
        k = k.reshape(src_len, bsz * self.nhead, head_dim)

        assert v.size(-1) % self.nhead == 0, "value's embed_dim must be divisible by the number of heads"
        head_dim = v.size(-1) // self.nhead
        v = v.reshape(src_len, bsz * self.nhead, head_dim)

        attn_output, attn_output_weights = self.attention_layer(
            q, k, v, attn_mask=attn_mask, bias_k=bias_k, bias_v=bias_v
        )
        attn_output = attn_output.reshape(tgt_len, bsz, -1)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(-3, -2)

        return attn_output, attn_output_weights

#taken from the LucidRain perciever IO github:
def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = th.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * th.pi
    x = th.cat([x.sin(), x.cos()], dim = -1)
    x = th.cat((x, orig_x), dim = -1)
    return x

class P_SIGLU(nn.Module):
    def __init__(self,dim):
        super(P_SIGLU, self).__init__()
        self.sig_weight = nn.Parameter(th.ones(dim,requires_grad=True))
        self.sig_bias = nn.Parameter(th.zeros(dim,requires_grad=True))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        return self.sigmoid(x*self.sig_weight+self.sig_bias)*x


if __name__ == '__main__':
    pass