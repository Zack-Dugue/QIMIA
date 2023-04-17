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
    def __init__(self, k_dim, n_heads, v_dim=None,w0=False,norm_query=True, end_norm = True, attention = None):
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
        if end_norm == True:
            self.end_norm = nn.LayerNorm(v_dim)
        else:
            self.end_norm = nn.Identity()

    def multihead_reshape(self,x):
        clz = x.size()[-1]
        assert(clz % self.n_heads == 0)
        bsz = x.size()[0]
        new_shape = list(x.size())
        new_shape[0] = bsz * self.n_heads
        new_shape[-1] = clz // self.n_heads
        try:
            x = x.view(new_shape)
        except RuntimeError:
            x = x.contiguous().view(new_shape)
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
        Q = self.multihead_reshape(Q.expand([K.size(0) , 1 , K.size(2)]))
        K = self.multihead_reshape(K)
        V = self.multihead_reshape(V)

        A = self.attention(Q, K, V)[0]
        A = self.multihead_unshape(A)
        if self.w0 is not None:
            A = self.w0(A)
        A = self.end_norm(A)
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


#This is like normal QIMIA, except there is no input projection,
# and it's expected that the inputs are two Tensors, representing Values and Keys.
class QIMIA2(nn.Module):
    def __init__(self,blocks, key_proj,value_proj,attention, value_out_transform=nn.Identity(), key_out_transform=nn.Identity(), layer_wise = False, through = False):
        """

        :param blocks: - an nn.module list of blocks
        :param key_dim_size: - the size of the key dimension
        :param key_proj: - the size of the key dimension
        :param block_input_dim: - the dimensions of the input to the block. Should NOT include the batch dim.
        currently only supports values of unchanging dimension.
        """
        super(QIMIA2, self).__init__()
        if layer_wise:
            self.layer_wise = layer_wise
        else:
            self.layer_wise = None

        self.through = through
        self.blocks = blocks
        self.key_proj = key_proj
        self.value_proj = value_proj
        self.flatten_to_token = nn.Flatten(0,-2)
        self.key_out_transform = key_out_transform
        self.value_out_transform = value_out_transform
        self.QIAttention = attention

    def forward(self, keys,values,block_input_dim):
        """

        :param keys:
        :param values:
        :param block_input_dim: A list containing the dimension which the block
                                input should be converted to. Not including the batch dimension
        :return:
        """
        for block , attention , key_proj, value_proj in zip(self.blocks, self.QIAttention, self.key_proj,self.value_proj):
            A = attention((keys, values))

            if not self.layer_wise:
                o = block(A.view([-1] + block_input_dim))
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


