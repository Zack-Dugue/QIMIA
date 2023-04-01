import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt



#Tokenwise QIMIA
# Through Block not yet implemented.
# Multihead Attention Not Yet Implemented.
class LearnedQueryAttention(nn.Module):
    # assuming this shape for tokens:
      # batch - other dims - channels
    def __init__(self, k_dim, n_heads, v_dim=None,w0=False,norm_query=True, attention = None):
        super(LearnedQueryAttention, self).__init__()
        self.q = th.randn(k_dim,requires_grad=True)
        self.norm_query = norm_query
        assert(k_dim % n_heads == 0)
        if w0:
            assert(v_dim is not None)
            self.w0 = nn.Linear(v_dim,v_dim)
        else:
            self.w0 = None
        if attention is not None:
            self.attention = attention
        else:
            self.attention = tt.nn.ScaledDotProduct(batch_first = True)

        self.n_heads = n_heads

    def multihead_reshape(self,x):
        clz = x.size()[-1]
        assert(clz % self.n_heads == 0)
        bsz = x.size()[0]
        new_shape = list(x.size())
        new_shape[0] = bsz * self.n_heads
        new_shape[-1] = clz // self.n_heads
        x = x.view(new_shape)
        return x

    def multihead_unshape(self,x):
        clz = x.size()[-1]
        bsz = x.size()[0]
        new_shape = list(x.size())
        new_shape[0] = bsz // self.n_heads
        new_shape[-1] = clz * self.n_heads
        x = x.view(new_shape)
        return x


    def forward(self, x):
        assert(type(x) == tuple)
        K , V = x
        if self.norm_query:
            Q = self.q/th.linalg.norm(self.q)
        else:
            Q = self.q
        # Q = self.q.repeat(K.size())
        Q = self.multihead_reshape(Q.repeat([K.size()[0], 1,1]))
        K = self.multihead_reshape(K)
        V = self.multihead_reshape(V)

        A = self.attention(Q, K, V)[0]
        A = self.multihead_unshape(A)
        if self.w0 is not None:
            A = self.w0(A)
        return A


class LearnedQueryAttention2(nn.Module):
    # assuming this shape for tokens:
      # batch - other dims - channels
    def __init__(self, k_dim, n_heads, v_dim=None,w0=False,norm_query=True):
        super(LearnedQueryAttention2, self).__init__()
        self.norm_query = norm_query
        assert(k_dim % n_heads == 0)
        if w0:
            assert(v_dim is not None)
            self.w0 = nn.Linear(v_dim,v_dim)
        else:
            self.w0 = None
        self.conv = nn.Conv2d(k_dim,n_heads,1,groups=n_heads)

        self.n_heads = n_heads
        self.softmax = nn.Softmax()

    def multihead_reshape(self,x):
        clz = x.size()[-1]
        assert(clz % self.n_heads == 0)
        bsz = x.size()[0]
        new_shape = list(x.size())
        new_shape[0] = bsz * self.n_heads
        new_shape[-1] = clz // self.n_heads
        x = x.view(new_shape)
        return x

    def multihead_unshape(self,x):
        clz = x.size()[-1]
        bsz = x.size()[0]
        new_shape = list(x.size())
        new_shape[0] = bsz // self.n_heads
        new_shape[-1] = clz * self.n_heads
        x = x.view(new_shape)
        return x


    def forward(self, x):
        assert(type(x) == tuple)
        K , V = x

        # Q = self.q.repeat(K.size())
        attention_map = self.conv(K.T).T
        attention_map = attention_map.contiguous().view([K.size()[0] * self.n_heads,K.size()[1]])
        attention_map = self.softmax(attention_map)
        V = self.multihead_reshape(V)
        A = th.matmul(attention_map,V)
        A = self.multihead_unshape(A)
        if self.w0 is not None:
            A = self.w0(A)
        return A


# For now this only supports properly flattening tokens, for dot product attention.
# in theory though it could support other attention.
class QIMIA(nn.Module):
    def __init__(self,blocks, key_proj,value_proj, input_key_proj, input_value_proj,attention, value_out_transform=nn.Identity(), key_out_transform=nn.Identity(), layer_wise = False, through = False):
        """

        :param blocks: - an nn.module list of blocks
        :param key_dim_size: - the size of the key dimension
        :param key_proj: - the size of the key dimension
        currently only supports values of unchanging dimension.
        """
        super(QIMIA, self).__init__()
        if layer_wise:
            self.layer_wise = layer_wise
        else:
            self.layer_wise = None

        self.through = through
        self.blocks = blocks
        self.key_proj = key_proj
        self.input_key_proj = input_key_proj
        self.value_proj = value_proj
        self.input_value_proj = input_value_proj
        self.flatten_to_token = nn.Flatten(0,-2)
        self.key_out_transform = key_out_transform
        self.value_out_transform = value_out_transform
        self.QIAttention = attention

    def forward(self, x):
        if not self.layer_wise:
            keys = th.stack([self.flatten_to_token(self.input_key_proj(x))], 1)
            v0 = self.input_value_proj(x)
            v0_shape = v0.size()
            values = th.stack([self.flatten_to_token(v0)],1)
        else:
            values = th.stack([x],1)
        for block , attention , key_proj, value_proj in zip(self.blocks, self.QIAttention, self.key_proj,self.value_proj):
            A = attention((keys, values))

            if not self.layer_wise:
                o = block(A.view(v0_shape))
                o = self.flatten_to_token(o)
            else:
                o = block(A)

            if self.through:
                o = o + A
            k = key_proj(o)
            v = value_proj(o)
            values = th.cat([values, th.unsqueeze(v,1)] , 1)
            keys = th.cat([keys,th.unsqueeze(k,1)], 1)
        keys = self.key_out_transform(keys)
        values = self.value_out_transform(values)
        return keys , values




class ind_KIMIA(nn.Module):
    def __init__(self, block,  key_dim_size, num_iters, t_resolution, encode=nn.Identity(),decode=nn.Identity()):
        super(ind_KIMIA, self).__init__()
        self.block = block
        self.query_proj = nn.LazyLinear(key_dim_size + t_resolution)
        self.key_proj= nn.LazyLinear(key_dim_size  + t_resolution)
        self.value_proj = nn.LazyLinear()
        # input key projector:
        self.input_key_proj = nn.LazyLinear(key_dim_size)
        self.input_value_proj = nn.LazyLinear(key_dim_size)

        self.attention = tt.nn.ScaledDotProduct()
        self.encode = nn.Identity()
        self.decode = nn.Identity()
        self.num_iters = num_iters
        self.t_encoding = th.Tensor([th.pi * (1/2)**x for x in range(t_resolution)])

    def forward(self,x,keys=[],values=[]):
        x = self.encode(x)
        keys = th.stack(keys.append(self.input_key_proj(x)),1)
        values = th.stack(values.append(self.input_value_proj(x)),1)
        q = th.ones(keys.size())
        for t in range(self.num_iters-1):
            A = self.attention(q,keys,values)
            B = self.block(A,t=t)
            v = self.value_proj(B)
            k = self.key_proj(th.cat([B,th.sin(t*self.t_encoding)],-1),1)
            q = self.query_proj(B)
            values = th.cat([values, v],1)
            keys = th.cat([keys,k],1)

        A = self.attention(q, keys, values)
        B = self.block(A, t=self.num_blocks)
        out = self.decode(B)
        return out