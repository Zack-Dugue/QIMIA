import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
import math
import csv
from utils import ScaledDotProduct, get_heads

#
# #replace this with torch text multi attention head thingy
# class LearnedQueryAttention(nn.Module):
#     # assuming this shape for tokens:
#       # batch - other dims - channels
#     def __init__(self, k_dim, v_dim, n_heads, layer_num = None, softmax_affine= False, attn_bias=True):
#         """
#         :param k_dim:
#         :param n_heads:
#         :param v_dim:
#         :param w0:
#         :param norm_query:
#         :param end_norm:
#         :param attention:
#         :param diagnostic:
#         """
#         super(LearnedQueryAttention, self).__init__()
#         # self.log = log
#         # self.q = nn.Parameter(th.ones(k_dim,requires_grad=True)/th.linalg.norm(th.ones(k_dim)))
#         self.q = nn.Parameter(th.zeros(k_dim,requires_grad=True))
#         assert(k_dim % n_heads == 0)
#         assert(v_dim is not None)
#         self.w0 = nn.Linear(v_dim,v_dim)
#         self.w0.weight = nn.Parameter(th.eye(v_dim,requires_grad=True))
#         self.w0.bias = nn.Parameter(th.zeros(v_dim,requires_grad=True))
#
#         self.attention = ScaledDotProduct(batch_first = True)
#
#         self.n_heads = n_heads
#         self.end_norm = nn.LayerNorm(v_dim)
#
#         self.attn_bias = attn_bias
#         if attn_bias:
#             self.v_bias = nn.Parameter(th.zeros(v_dim,requires_grad=True))
#             self.k_bias = nn.Parameter(th.zeros(k_dim,requires_grad=True))
#         else:
#             self.v_bias = None
#             self.k_bias = None
#         #
#         # self.log = log
#         # if log:
#         #     self.Slog = []
#         #     self.Rlog = []
#         #     self.A_wlog =[]
#         #     self.norm_qlog = []
#         #     self.norm_Alog = []
#         # self.print_log = print_log
#
#     def multihead_reshape(self,x):
#         clz = x.size()[-1]
#         assert(clz % self.n_heads == 0)
#         bsz = x.size()[0]
#         new_shape = list(x.size())
#         new_shape[0] = bsz * self.n_heads
#         new_shape[-1] = clz // self.n_heads
#         x = x.contiguous().view(new_shape)
#         return x
#
#     def multihead_unshape(self,x):
#         clz = x.size()[-1]
#         bsz = x.size()[0]
#         new_shape = list(x.size())
#         new_shape[0] = bsz // self.n_heads
#         new_shape[-1] = clz * self.n_heads
#         x = x.view(new_shape)
#         return x
#
#
#     def forward(self, keys,values):
#         K , V = keys,values
#         Q = self.q
#         # Q = self.q.repeat(K.size())
#         if self.attn_bias:
#             v_bias = self.multihead_reshape(self.v_bias.expand([V.size(0) , 1 , V.size(2)]))
#             k_bias = self.multihead_reshape(self.k_bias.expand([K.size(0) , 1 , K.size(2)]))
#             bias_k = th.permute(k_bias,[1,0,2])
#             bias_v = th.permute(v_bias,[1,0,2])
#         else:
#             bias_k = None
#             bias_v = None
#         Q = self.multihead_reshape(Q.expand([K.size(0) , 1 , K.size(2)]))
#         K = self.multihead_reshape(K)
#         V = self.multihead_reshape(V)
#         A, A_w = self.attention(Q, K, V,bias_k = bias_k, bias_v = bias_v)
#
#
#
#         A = self.multihead_unshape(A)
#
#         A = self.w0(A)
#         A = self.end_norm(A)
#         # A = th.sum(values, dim=1,keepdim=True)
#
#         return A
#
#     def diagnostic_run(self,A,A_w):
#         """
#         Performs a diagnostic analysis of the run and saves it to a buffer.
#         :param A: the actual output of the LQA after the multihead unshape
#         :param A_w: attention weights of the shape [Num_Heads * Bsz,NumLayers,1]
#         :return:
#         """
#         A_w = th.squeeze(A_w)
#         if len(A_w.size()) == 1:
#             A_w = th.unsqueeze(A_w,1)
#         num_layers = A_w.size(1)
#         S = th.mean(th.sum(-th.log(A_w)*A_w,dim=1),dim=0)/math.log(num_layers)
#         norm_q = th.linalg.norm(self.q)
#         variance_q = th.var(self.q,0)
#         norm_A = th.mean(th.linalg.norm(A))
#         u = th.mean(A,0)
#         u = u.expand(A.size())
#         R = th.mean(th.norm(A-u)**2)
#         # print(f"S = {S} , R = {R} , A_w= {th.mean(A_w,0)},  norm_q = {norm_q} , norm_A = {norm_A}")
#         # print(f"Q = {self.q}")
#         if self.print_log:
#             print(f" A_w= {th.mean(A_w,0)}, norm_q = {norm_q}, variance_q = {variance_q}")
#         self.Slog.append(float(th.nan_to_num(S)))
#         self.Rlog.append(float(R))
#         self.A_wlog.append(th.mean(A_w,0).tolist())
#         self.norm_qlog.append(float(norm_q))
#         self.norm_Alog.append(float(norm_A))
#     def get_log(self):
#         return {"S value" : self.Slog, "R value" : self.Rlog, "attn weight" : self.A_wlog, "Norm of Query" : self.norm_qlog, "Norm of Output Attention" :  self.norm_Alog}

class LearnedQueryAttention(nn.Module):
    def __init__(self, key_dim, embed_dim, n_heads, layer_num = None, softmax_affine= False, attn_bias=True):
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
        self.norm = nn.LayerNorm(embed_dim)

        self.n_heads = n_heads
        self.QConv = nn.Conv1d(key_dim, n_heads,1, bias=False, groups=n_heads)
        self.QConv.weight = nn.Parameter(th.ones_like(self.QConv.weight,requires_grad=True))
        if layer_num is not None:
            self.QConv.weight = th.ones_like(self.QConv.weight)*layer_num
        self.w0 = nn.Linear(embed_dim, embed_dim)
        self.w0.weight = nn.Parameter(th.eye(embed_dim, requires_grad=True))
        self.w0.bias = nn.Parameter(th.zeros(embed_dim, requires_grad=True))
        self.attn_bias = attn_bias

        if attn_bias:
            self.v_bias = nn.Parameter(th.zeros(embed_dim,requires_grad=True))
            self.k_bias = nn.Parameter(th.randn(key_dim,requires_grad=True)/(key_dim**(.5)))
        self.softmax_affine = softmax_affine
        if softmax_affine:
            self.softmax_affine = True
            assert(layer_num is not None)
            self.softmax_weight = nn.Parameter(th.ones([layer_num+attn_bias,n_heads],requires_grad=True))
            self.softmax_bias = nn.Parameter(th.zeros([layer_num+attn_bias,n_heads],requires_grad=True))

    def instantiate_softmax_affine(self, layer_num):
        self.softmax_affine = True
        self.softmax_weight = nn.Parameter(th.ones([self.n_heads,layer_num + self.attn_bias], requires_grad=True))
        self.softmax_bias = nn.Parameter(th.zeros([self.n_heads,layer_num + self.attn_bias], requires_grad=True))
        self.QConv.weight = nn.Parameter(th.ones_like(self.QConv.weight,requires_grad=True)*layer_num)

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

        keys = th.permute(keys,[0,2,1])

        A_w = self.QConv(keys)

        #Aw dims (batch, token, head)
        if self.softmax_affine:
            A_w = A_w * self.softmax_weight + self.softmax_bias
        A_w = F.softmax(A_w,2)
        A_w = th.flatten(A_w,0,1)
        A_w = th.unsqueeze(A_w,1)
        V = self.multihead_reshape(values)
        A = th.matmul(A_w,V)
        A = self.multihead_unshape(A)
        A = self.w0(A)
        A = self.norm(A)
        return A

    def get_log(self):
        return None


#TODO switch everything from batch first to putting the sequences dimension first.
class BaseBlock(nn.Module):
    def __init__(self, key_dim, embed_dim, num_lqa_heads=None, attn_bias=False,softmax_affine=False):
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
    def __init__(self, key_dim, embed_dim, num_lqa_heads=None,softmax_affine = False, attn_bias=False, diagnostic=False):
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
    def __init__(self,blocks, set_up_softmax_affine = False):
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

    def get_logs(self,path = None):
        logs_dict = {}
        for i, block in enumerate(self.blocks):
            if issubclass(block.__class__,OutputBlock) or issubclass(block.__class__,BaseBlock):
                logs = block.LQA.get_log()
                logs_dict.update({f"block_{i}":logs})
        if path == None:
            return logs_dict
        else:
            for (name,block_log) in logs_dict.items():
                f = open(path + "/" + name + ".csv", "w",newline='')
                writer = csv.writer(f)
                keys = list(block_log.keys())
                new_keys = []
                for key in keys:
                   if list ==type(block_log[key][0]):
                       for i in range(len(block_log[key][0])):
                        new_keys.append(f"{key}_{i}")
                   else:
                       new_keys.append(key)
                writer.writerow(new_keys)
                log_this = [[list(block_log.values())[j][i] for j in range(len(list(block_log.values())))] for i in range(len(list(block_log.values())[0]))]
                new_log_this = []
                for row in log_this:
                    row_list = []
                    for element in row:
                        if list == type(element):
                            for i in range(len(element)):
                                row_list.append(element[i])
                        else:
                            row_list.append(element)
                    new_log_this.append(row_list)
                writer.writerows(new_log_this)
        print("Done Logging")



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