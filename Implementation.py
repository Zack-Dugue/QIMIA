import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
import math
import csv
from utils import ScaledDotProduct, get_heads


class LearnedQueryAttention(nn.Module):
    def __init__(self, key_dim, embed_dim, n_heads, layer_num = None, softmax_affine= False, attn_bias=True, log = True):
        """
        :param key_dim: the key dimension
        :param embed_dim: the value dimension
        :param num_lqa_heads: the number of Learned Query Attention Heads
        :param Norm: None implies no norm is Used. 'pre' applies the norm after the LQA.
                    'post value' applies the norm to the values.
                    'post key' applies the norm to the key.
                    'post both' applies the norm to the values and the keys.
        """
        super(LearnedQueryAttention, self).__init__()
        if n_heads == None:
            # implement some way of inteligently selecting a reasonable
            # number of heads based on the key_dim
            n_heads = 4
        self.log = log
        self.norm = nn.LayerNorm(embed_dim)

        self.n_heads = n_heads
        if layer_num is not None:
            self.query = nn.Parameter(th.ones([self.n_heads, 1, key_dim // self.n_heads],requires_grad=True)*layer_num)
        else:
            self.query = nn.Parameter(th.zeros([self.n_heads, 1, key_dim // self.n_heads],requires_grad=True))


        self.attn_bias = attn_bias

        if attn_bias:
            self.v_bias = nn.Parameter(th.zeros(embed_dim,requires_grad=True))
            self.k_bias = nn.Parameter(th.randn(key_dim,requires_grad=True)/(key_dim**(.5)))
        self.softmax_affine = softmax_affine
        if softmax_affine:
            self.softmax_affine = True
            assert(layer_num is not None)
            self.softmax_weight = nn.Parameter(th.ones([n_heads,1,layer_num+attn_bias],requires_grad=True))
            self.softmax_bias = nn.Parameter(th.zeros([n_heads,1,layer_num+attn_bias],requires_grad=True))

    def instantiate_softmax_affine(self, layer_num):
        self.softmax_affine = True
        self.softmax_weight = nn.Parameter(th.ones([self.n_heads,1,layer_num + self.attn_bias], requires_grad=True))
        self.softmax_bias = nn.Parameter(th.zeros([self.n_heads,1,layer_num + self.attn_bias], requires_grad=True))
        # self.query = nn.Parameter(th.ones_like(self.query,requires_grad=True)*layer_num)

    def multihead_reshape(self,x):
        clz = x.size()[-1]
        assert(clz % self.n_heads == 0)
        bsz = x.size()[0]
        new_shape = list(x.size())
        new_shape[0] = bsz * self.n_heads
        new_shape[-1] = clz // self.n_heads
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

    #There's an issue with keeping everything contiguous here.
    # We have to do our convolution over the token dimension.
    # And we want it to output (batch, token, head)
    # But then we have to flatten the head dimension into the batch dimension.
    # Which gives us some contiguousness issues.
    def forward(self,keys:th.Tensor,values : th.Tensor):
        #dims of inputs: (batch, token, channel)
        if self.attn_bias:

            keys = th.cat([keys, self.k_bias.expand([keys.size(0), 1 , self.k_bias.size(0)])],1)
            values = th.cat([values, self.v_bias.expand([values.size(0), 1 , self.v_bias.size(0)])],1)
        else:
            # This is necessary due to issues with the compute graph. Basically, since we use a view
            # of the keys.
            # TODO fix this at some point.
            keys = keys*1
        Q = self.query.repeat([keys.size(0),1,1])
        K = self.multihead_reshape(keys)

        A_w = th.bmm(Q,th.permute(K,[0,2,1]))

        #Aw dims (batch, token, head)
        if self.softmax_affine:
            A_w = A_w * self.softmax_weight.repeat([keys.size(0),1,1]) + self.softmax_bias.repeat([keys.size(0),1,1])
        A_w = F.softmax(A_w,2)
        A_w = th.flatten(A_w,0,1)
        A_w = th.unsqueeze(A_w,1)
        V = self.multihead_reshape(values)
        A = th.matmul(A_w,V)
        A = self.multihead_unshape(A)
        # A = self.w0(A)
        if self.log:
            self.compute_log(A_w,A)

        A = self.norm(A)

        return A
    def compute_log(self,A_w,A):
        with th.no_grad():
            self.H = th.mean(th.sum(A_w*th.log(A_w),2))/math.log(A_w.size(2))
            self.R = th.mean(A)
            self.S = th.mean(th.var(A,dim=0))
            self.q_norm = th.linalg.vector_norm(th.flatten(self.query))

    def get_log(self):
        return {'H':self.H, 'R':self.R, 'S':self.S, 'q_norm':self.q_norm}


#TODO switch everything from batch first to putting the sequences dimension first.
class BaseBlock(nn.Module):
    def __init__(self, key_dim, embed_dim, num_lqa_heads=None, attn_bias=True,softmax_affine=False):
        """
        :param key_dim: the key dimension
        :param embed_dim: the value dimension
        :param num_lqa_heads: the number of Learned Query Attention Heads
        :param Norm: None implies no norm is Used.
                    'pre' applies the norm after the LQA.
                    'pre + key' applies the norm after the LQA and then also to the key.
                    'post value' applies the norm to the values.
                    'post key' applies the norm to the key.
                    'post both' applies the norm to the values and the keys.
        """
        super(BaseBlock,self).__init__()
        self.output_depth = 1
        if num_lqa_heads == None:
            # implement some way of inteligently selecting a reasonable
            # number of heads absed on the key_dim
            num_lqa_heads= get_heads(key_dim)

        self.LQA  = LearnedQueryAttention(key_dim,embed_dim, num_lqa_heads,attn_bias = attn_bias, softmax_affine =softmax_affine)
        # self.LQA = LQASimple(key_dim,num_lqa_heads,v_dim=embed_dim,w0=True)
    def block(self,A,**kwargs):
        pass
    def forward(self, keys , values, layer_num = None, append=True,check_dim = False, **aux):
        A = th.squeeze(self.LQA(keys[:,:layer_num,:],values[:,:layer_num,:]))
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
        # key = th.zeros_like(key)
        # value = th.randn_like(value)
        if layer_num is not None:
            keys[:,layer_num,:] = key
            values[:,layer_num,:] = value
            return keys , values

        if append == True:
            keys = th.cat([keys,th.unsqueeze(key,1)],1)
            values = th.cat([values,th.unsqueeze(value,1)],1)
            return keys , values
        else:
            return key , value

class InitializerBlock(nn.Module):
    def __init__(self,output_depth=1):
        super(InitializerBlock,self).__init__()
        # Output Depth is just the number of keys or values outputted by the Block.
        # For Base Blocks this is always 1, but for Initializer Blocks it can vary.
        self.output_depth = output_depth

    def block(self,*args):
        pass
    def forward(self,*args,**kwargs):
        keys , values = self.block(*args,**kwargs)
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
        self.output_depth = 0
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
    def __init__(self, key_dim, embed_dim, num_lqa_heads=None,softmax_affine = False, attn_bias=True, diagnostic=False):
        super(OutputBlock, self).__init__()
        if num_lqa_heads == None:
            # implement some way of inteligently selecting a reasonable
            # number of heads absed on the key_dim
            num_lqa_heads = get_heads(key_dim)
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.LQA = LearnedQueryAttention(key_dim, embed_dim, num_lqa_heads,softmax_affine=softmax_affine,attn_bias=attn_bias)
        # self.LQA = LQASimple(key_dim, num_lqa_heads, v_dim=embed_dim, w0=True,diagnostic=diagnostic)
    def block(self,A, **kwargs):
        pass
    def forward(self,keys,values, layer_num = None, **aux):
        A = th.squeeze(self.LQA(keys,values))
        if aux==None:
            output = self.block(A)
        else:
            output = self.block(A,**aux)

        return output

class QIMIA_Sequential(nn.Module):
    def __init__(self,blocks, set_up_softmax_affine = True):
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
        if set_up_softmax_affine:
            assert(self.has_initializer)
            layer_num_counter = 0
            for block in self.blocks:
                if issubclass(block.__class__, InitializerBlock):
                   layer_num_counter += block.output_depth
                   continue
                else:
                    block.LQA.instantiate_softmax_affine(layer_num_counter)
                    if not issubclass(block.__class__, OutputBlock):
                        layer_num_counter += block.output_depth

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
            start_keys, start_values = self.blocks[0](*x,**aux_list[0])
        else:
          start_keys , start_values = self.blocks[0](*x)
        num_start = start_values.size(1)
        num_other_layers = len(self.blocks)-1-(issubclass(self.blocks[-1].__class__,OutputBlock))
        values = th.zeros(start_values.size(0), num_start + num_other_layers , start_values.size(2),device=start_values.device)
        values[:,0:num_start,:] = start_values
        keys = th.zeros(start_keys.size(0), num_start + num_other_layers , start_keys.size(2),device=start_values.device)
        keys[:,0:num_start,:] = start_keys
        x = (keys,values)
        if len(aux_list)!=0:
            assert(len(aux_list) == len(self.blocks))
            for(i, (block,aux)) in enumerate(zip(self.blocks[1:],aux_list[1:])):
                    x = block(*x, **aux,layer_num = num_start+i)
        else:
            for (i,block) in enumerate(self.blocks[1:]):
                x = block(*x,layer_num = num_start+i)
        return x

    def __len__(self):
        return len(self.blocks)

    def get_logs(self, avg_logs = True):
        logs_dict = {'H' : 0 , 'R' : 0 , 'S' : 0 , 'q_norm' : 0}
        for i, block in enumerate(self.blocks):
            if issubclass(block.__class__,OutputBlock) or issubclass(block.__class__,BaseBlock):
                logs = block.LQA.get_log()
                logs_dict['H'] += logs['H']
                logs_dict['R'] += logs['R']
                logs_dict['S'] += logs['S']
                logs_dict['q_norm'] += logs['q_norm']
        logs_dict['H'] /= len(self.blocks)
        return logs_dict






class QIMIA_Parallel(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        assert(type(blocks) == list or type(blocks) == nn.ModuleList
               or issubclass(blocks.__class__,nn.ModuleList))
        self.blocks = blocks
        self.has_initializer = False
        self.output_depth = len(blocks)
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