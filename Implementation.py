import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
import math


class LearnedQueryAttention(nn.Module):
    # assuming this shape for tokens:
      # batch - other dims - channels
    def __init__(self, k_dim, n_heads, v_dim=None,w0=False,norm_query=False, end_norm = False, attention = None,diagnostic = True):
        """

        :param k_dim:
        :param n_heads:
        :param v_dim:
        :param w0:
        :param norm_query:
        :param end_norm:
        :param attention:
        :param diagnostic:
        """
        super(LearnedQueryAttention, self).__init__()
        self.diagnostic = diagnostic
        self.q = nn.Parameter(th.randn(k_dim,requires_grad=True))
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
        #TODO is this the problem?
        # Is this jumbling up the information somehow?
        clz = x.size()[-1]
        bsz = x.size()[0]
        new_shape = list(x.size())
        new_shape[0] = bsz // self.n_heads
        new_shape[-1] = clz * self.n_heads
        x = x.view(new_shape)
        return x


    def forward(self, keys,values):
        K , V = keys,values
        if self.norm_query:
            Q = self.q/th.linalg.norm(self.q)
        else:
            Q = self.q
        # Q = self.q.repeat(K.size())
        Q = self.multihead_reshape(Q.expand([K.size(0) , 1 , K.size(2)]))
        K = self.multihead_reshape(K)
        V = self.multihead_reshape(V)

        A, A_w = self.attention(Q, K, V)
        A = self.multihead_unshape(A)
        if self.w0 is not None:
            A = self.w0(A)
        if self.diagnostic:
            self.diagnostic_run(A,A_w)
        A = self.end_norm(A)
        return A

    def diagnostic_run(self,A,A_w):
        """
        Performs a diagnostic analysis of the run and saves it to a buffer.
        :param A: the actual output of the LQA after the multihead unshape
        :param A_w: attention weights of the shape [Num_Heads * Bsz,NumLayers,1]
        :return:
        """
        A_w = th.squeeze(A_w)
        if len(A_w.size()) == 1:
            A_w = th.unsqueeze(A_w,1)
        num_layers = A_w.size(1)
        S = th.mean(th.sum(-th.log(A_w)*A_w,dim=1),dim=0)/math.log(num_layers)
        u = th.mean(A_w,0)
        u = u / th.linalg.norm(u,1)
        R = -F.kl_div(A_w, u.expand([A_w.size(0),u.size(0)]))
        norm_q = th.linalg.norm(self.q)
        norm_A = th.mean(th.linalg.norm(A))
        u = th.mean(A,0)
        u = u.expand(A.size())
        R = th.mean(th.norm(A-u)**2)
        print(f"S = {S} , R = {R} , A_w= {th.mean(A_w,0)},  norm_q = {norm_q} , norm_A = {norm_A}")
        # print(f"Q = {self.q}")
        # print(f" A_w= {th.mean(A_w,0)}")

class LQASimple(nn.Module):
    """This model is INEFFICIENT , contains uncessary projection layers.
    IT's soley a debugging tool"""
    def __init__(self, k_dim, n_heads, v_dim=None,w0=False,norm_query=False, end_norm = True, attention = None,diagnostic = False):
        super().__init__()
        self.diagnostic = diagnostic
        self.attention = nn.MultiheadAttention(v_dim,n_heads,vdim=v_dim,kdim=k_dim,batch_first=True)
        # nn.MultiheadAttention()
        self.norm = nn.LayerNorm(v_dim)
        self.q = th.ones(k_dim)/th.linalg.norm(th.ones(k_dim))
    def forward(self,keys,values):
        K , V = keys,values
        # if self.norm_query:
        #     Q = self.q/th.linalg.norm(self.q)
        # else:
        #     Q = self.q
        # Q = self.q.repeat(K.size())
        Q = self.q
        Q = Q.expand([K.size(0) , 1 , K.size(2)])


        A, A_w = self.attention(Q, K, V)
        if self.diagnostic:
            self.diagnostic_run(A,A_w)
        return self.norm(A)

    def diagnostic_run(self,A,A_w):
        """
        Performs a diagnostic analysis of the run and saves it to a buffer.
        :param A: the actual output of the LQA after the multihead unshape
        :param A_w: attention weights of the shape [Num_Heads * Bsz,NumLayers,1]
        :return:
        """
        A_w = th.squeeze(A_w)
        num_layers = A_w.size(1)
        S = th.mean(th.sum(-th.log(A_w)*A_w,dim=1),dim=0)/math.log(num_layers)
        norm_q = th.linalg.norm(F.linear(self.q , self.attention.q_proj_weight, bias=None))
        norm_A = th.mean(th.linalg.norm(A))
        print(f"S = {S} , norm_q = {norm_q} , norm_A = {norm_A}")
class BaseBlock(nn.Module):
    def __init__(self, key_dim, embed_dim, num_lqa_heads=None, norm = 'pre'):
        """

        :param key_dim: the key dimension
        :param embed_dim: the value dimension
        :param num_lqa_heads: the number of Learned Query Attention Heads
        :param Norm: None implies no norm is Used. 'pre' applies the norm after the LQA.
                    'post value' applies the norm to the values.
                    'post key' applies the norm to the key.
                    'post both' applies the norm to the values and the keys.
        """
        super(BaseBlock,self).__init__()
        if num_lqa_heads == None:
            # implement some way of inteligently selecting a reasonable
            # number of heads absed on the key_dim
            num_lqa_heads = 1
        if norm == 'pre': lqa_end_norm = True
        else: lqa_end_norm = True
        if norm == 'post value':
            self.value_norm = nn.LayerNorm(embed_dim)
            self.key_norm = nn.Identity()
        if norm == 'post key':
            self.value_norm = nn.Identity()
            self.key_norm = nn.BatchNorm1d(embed_dim)
        elif norm == 'post both':
            self.value_norm = nn.LayerNorm(embed_dim)
            self.key_norm = nn.LayerNorm(key_dim)
        else:
            self.key_norm = nn.Identity()
            self.value_norm = nn.Identity()
        self.LQA  = LearnedQueryAttention(key_dim,num_lqa_heads,v_dim=embed_dim,w0=True,end_norm=lqa_end_norm)
        # self.LQA = LQASimple(key_dim,num_lqa_heads,v_dim=embed_dim,w0=True)
    def block(self,A,**kwargs):
        pass
    def forward(self, keys , values, append=True,check_dim = False, **aux):
        A = th.squeeze(self.LQA(keys,values))
        if aux is not None:
            key , value = self.block(A,**aux)
        else:
            key , value = self.block(A)
        if len(key.size()) != 2:
            if check_dim:
                raise ValueError(f"Key dimensionality of {key.size()} is invalid"
                                 f"as 2 dimensions were expected in the key output"
                                 f"and {len(key.size())} were recieved")
            key = th.flatten(key,0,-2)
        if len(value.size()) != 2:
            if check_dim:
                raise ValueError(f"Value dimensionality of {value.size()} is invalid"
                                 f"as 2 dimensions were expected in value key output"
                                 f"and {len(value.size())} were recieved")
            value = th.flatten(value,0,-2)
        key = self.key_norm(key)
        # key = th.zeros_like(key)
        value = self.value_norm(value)
        # value = th.randn_like(value)
        if append == True:
            keys = th.cat([keys,th.unsqueeze(key,1)],1)
            values = th.cat([values,th.unsqueeze(value,1)],1)
            return keys , values
        else:
            return key , value

class InitializerBlock(nn.Module):
    def __init__(self):
        super(InitializerBlock,self).__init__()

    def block(self,*args):
        pass
    def forward(self,*args):
        keys , values = self.block(*args)
        if type(keys) in [tuple, list] and type(values) in [tuple,list]:
            keys = th.stack(keys,1)
            values = th.stack(values,1)
            return keys , values
        else:
            keys = th.unsqueeze(keys,1)
            values = th.unsqueeze(values,1)

        return keys , values

class TransformBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def block(self,keys,values,check_dim = False,**kwargs):
        pass
    def forward(self,keys : th.Tensor,values : th.Tensor,**aux):
        old_token_len = keys.size(1)
        if aux == None:
            keys,values = self.block(keys,values)
        else:
            keys, values = self.block(keys,values,*aux)
        assert(len(keys.size()) == 3)
        assert(len(values.size()) == 3)
        assert(old_token_len == keys.size(1) == values.size(1))

        return keys,values

class OutputBlock(nn.Module):
    def __init__(self, key_dim, embed_dim, num_lqa_heads=None,diagnostic=False):
        super(OutputBlock, self).__init__()
        if num_lqa_heads == None:
            # implement some way of inteligently selecting a reasonable
            # number of heads absed on the key_dim
            num_lqa_heads = 8
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.LQA = LearnedQueryAttention(key_dim, num_lqa_heads, v_dim=embed_dim,end_norm=True, w0=True,diagnostic=diagnostic)
        # self.LQA = LQASimple(key_dim, num_lqa_heads, v_dim=embed_dim, w0=True,diagnostic=diagnostic)
    def block(self,A, **kwargs):
        pass
    def forward(self,keys,values,**aux):
        A = self.LQA(keys,values)
        if aux==None:
            output = self.block(A)
        else:
            output = self.block(A,**aux)

        return output

class QIMIA_Sequential(nn.Module):
    def __init__(self,blocks):
        super().__init__()
        assert(type(blocks) == list or type(blocks) == nn.ModuleList
               or issubclass(blocks,nn.ModuleList))
        self.blocks = nn.ModuleList(blocks)
        self.has_initializer = False
        for i , block in enumerate(blocks):
            if issubclass(block.__class__,InitializerBlock):
                if i == 0:
                    self.has_initializer = True
                    continue
                else:
                    raise ValueError("Initializer Block "
                                 "must be the first block in a QIMIA sequential")
            if issubclass(block.__class__,OutputBlock):
                if i == len(blocks)-1:
                    continue
                else:
                    raise ValueError("Output block"
                                     "must be the last block in a QIMIA sequential")
            if issubclass(block.__class__,TransformBlock) or issubclass(block.__class__,BaseBlock):
                continue
            else:
                raise ValueError("This block is not a subclass of the allowed"
                                 "block types")
    #Forward for the case where they are passed key and value and that's it

    def parameters(self):
        return self.blocks.parameters()
    def forward(self,*x,aux_list=[]):
        """

        :param x: the input
        :param aux: a list of dictionaries representing the auxillary kwargs for each block.
        :return:
        """
        if len(aux_list)!=0:
            assert(len(aux_list) == len(self.blocks))
            for block,aux in zip(self.blocks,aux_list):
                    x = block(*x, **aux)
        else:
            for block in self.blocks:
                x = block(*x)
        return x

    def __len__(self):
        return len(self.blocks)


class QIMIA_Parallel(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        assert(type(blocks) == list or type(blocks) == nn.ModuleList
               or issubclass(blocks.__class__,nn.ModuleList))
        self.blocks = blocks
        self.has_initializer = False
        for block in blocks:
            if not issubclass(block.__class__,BaseBlock):
                raise ValueError("Only BaseBlocks may be in QIMIA Parallel Modules")
    def forward(self,keys,values,aux=None):
        key_list = []
        value_list = []
        for block in self.blocks:
            if aux is None:
                key , value = block(keys,values,append=False)
                key_list.append(th.unsqueeze(key,1))
                value_list.append(th.unsqueeze(value,1))
        keys = th.cat([keys] + key_list)
        values = th.cat([values] + value_list)
        return keys , values

