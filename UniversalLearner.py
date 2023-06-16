import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
import torchvision as tv
from Implementation import LearnedQueryAttention
from FasterImplementation import BaseBlock, InitializerBlock, OutputBlock , TransformBlock, QIMIA_Sequential
from transformers import AutoTokenizer, PreTrainedTokenizer
from Vision import FF_Block
from utils import MultiheadAttentionContainer, fourier_encode
import math
import time


class SelfAttentionBlock(BaseBlock):
    def __init__(self, key_dim, embed_dim,num_latents, num_attention_heads=16):
        # Assumes the value dimension and key dimension of the encoder
        # are the same.
        super().__init__(key_dim, embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_attention_heads)
        self.value_act = nn.PReLU(embed_dim)
        self.value_project = nn.Linear(embed_dim, embed_dim)
        self.key_act = nn.PReLU(embed_dim)
        self.key_project = nn.Linear(embed_dim, key_dim)
        self.num_latents = num_latents

    def block(self, A):

        A = A.view(A.size(0) // self.num_latents, self.num_latents, A.size(1))
        o = self.self_attention(A, A, A)[0]
        o = th.flatten(o,0,1)
        key = self.key_act(o)
        key = self.key_project(key)

        value = self.value_act(o)
        value = self.value_project(value)

        return key, value

class CrossAttentionBlock(BaseBlock):
    def __init__(self,key_dim, embed_dim, cross_dim, num_latents ,num_cross_heads=1):
        # Assumes the value dimension and key dimension of the encoder
        # are the same.
        super().__init__(key_dim,embed_dim)
        self.num_latents = num_latents
        in_proj = tt.nn.InProjContainer(nn.Linear(embed_dim,cross_dim),nn.Linear(cross_dim,cross_dim) , nn.Linear(cross_dim,cross_dim))
        self.cross_attention = MultiheadAttentionContainer(num_cross_heads,in_proj,tt.nn.ScaledDotProduct(),nn.Linear(cross_dim,embed_dim),batch_first=True)
        self.value_act = nn.PReLU(embed_dim)
        self.value_project = nn.Linear(embed_dim,embed_dim)
        self.key_act = nn.PReLU(embed_dim)
        self.key_project = nn.Linear(embed_dim,key_dim)

    def block(self,A, cross_keys = None, cross_values = None):

        if cross_keys == None:
            raise ValueError("The keys for cross attention must be defined")
        if cross_values == None:
            raise ValueError("The values for cross attention must be defined")

        A = A.view(A.size(0)//self.num_latents, self.num_latents , A.size(1))
        o = self.cross_attention(A,cross_keys,cross_values)[0]
        o = th.flatten(o,0,1)
        key = self.key_act(o)
        key = self.key_project(key)

        value = self.value_act(o)
        value = self.value_project(value)

        return key, value

class InternalInitBlock(InitializerBlock):
    def __init__(self,key_dim, embed_dim, external_embed_dim,num_latents,T=None):
        super().__init__()
        #using Latent Specific Affine Transformations would probably be a good idea here.
        self.key_linear = nn.Linear(external_embed_dim,key_dim)
        self.value_linear = nn.Linear(external_embed_dim,embed_dim)
        self.key_PE = nn.Parameter(th.randn([num_latents,key_dim],requires_grad=True))
        self.value_PE = nn.Parameter(th.randn([num_latents, embed_dim],requires_grad=True))
        if T is not None:
            self.value_time = nn.Parameter(th.randn([T,embed_dim],requires_grad=True))
            self.key_time = nn.Parameter(th.randn([T,key_dim],requires_grad=True))
        self.num_latents = num_latents
    def block(self,A,t=None):
        key = self.key_linear(A)
        value = self.value_linear(A)
        key_PE = self.key_PE.repeat([A.size(0)//self.num_latents,1])
        value_PE = self.value_PE.repeat([A.size(0)//self.num_latents,1])
        if t is not None:
            value_t = self.value_time[t,:].repeat([A.size(0),1])
            key_t = self.key_time[t,:].repeat([A.size(0),1])
            return [key,key_t,key_PE] , [value, value_t, value_PE]

        return [key, key_PE] , [value, value_PE]

class HeadBlock(OutputBlock):
    def __init__(self,key_dim,embed_dim, output_dim, hidden_layer_dim = 2048):
        super().__init__(key_dim, embed_dim)
        self.L1 = nn.Linear(embed_dim, hidden_layer_dim)
        self.act1 = nn.GELU()
        self.L2 = nn.Linear(hidden_layer_dim,output_dim)
    def block(self,A):
        o = self.L1(A)
        o = self.act1(o)
        o = self.L2(o)
        return o

class QueryHeadBlock(OutputBlock):
    def __init__(self,key_dim,embed_dim, output_dim, hidden_layer_dim = 512):
        super().__init__(key_dim, embed_dim)
        #This is modeled after a GRU,
        self.init_h = nn.Parameter(th.randn(hidden_layer_dim,requires_grad=True))
        self.ext_q_norm = nn.LayerNorm(output_dim)
        self.Wir = nn.Linear(embed_dim, hidden_layer_dim)
        self.Whr = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.Wiz = nn.Linear(embed_dim, hidden_layer_dim)
        self.Whz = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.Win = nn.Linear(embed_dim, hidden_layer_dim)
        self.Whn  = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.Wout = nn.Linear(hidden_layer_dim, output_dim)
        self.Norm = nn.LayerNorm(hidden_layer_dim,eps=.1)
    def block(self,A , h =None):
        if h is None:
            h = self.init_h
        r = F.sigmoid(self.Wir(A) + self.Whr(h) )
        z = F.sigmoid(self.Wiz(A) + self.Whz(h))
        n = self.Norm(self.Win(A) + r*self.Whn(h))
        new_h = (th.ones_like(z) - z)*n + z*h
        out = self.Wout(new_h)
        return new_h, out

class SimpleQueryHeadBlock(OutputBlock):
    def __init__(self,key_dim,embed_dim, output_dim, hidden_layer_dim = 512):
        super().__init__(key_dim, embed_dim)
        self.init_h = nn.Parameter(th.randn(hidden_layer_dim//2,requires_grad=True))
        self.L0 = nn.Linear(embed_dim + hidden_layer_dim //2 , hidden_layer_dim)
        self.h_norm = nn.LayerNorm(hidden_layer_dim//2)
        self.act = nn.GELU()
        self.L1 = nn.Linear(hidden_layer_dim //2, output_dim)
    def block(self,A , h =None):
        if h is None:
            h = self.init_h
            h = h.expand([A.size(0), 1])

        h = self.h_norm(h)
        h_new = self.L0(th.cat([A,h],-1))
        h_new = self.act(h_new)
        out = self.L1(h_new)
        return h_new, out

class OutputHeadBlock(OutputBlock):
    def __init__(self,key_dim,embed_dim, output_dim,n_horizontal_heads = 8, hidden_layer_dim = 2048):
        super().__init__(key_dim, embed_dim)
        self.FF1 = nn.Linear(embed_dim, hidden_layer_dim)
        self.value_act = nn.PReLU(hidden_layer_dim)
        self.key_act = nn.PReLU(hidden_layer_dim)
        self.valueFF = nn.Linear(hidden_layer_dim, embed_dim)
        self.keyFF = nn.Linear(hidden_layer_dim, embed_dim)
        self.Horizontal_LQA = LearnedQueryAttention(embed_dim,n_horizontal_heads,v_dim=embed_dim)
        self.FF2 = nn.Linear(embed_dim, hidden_layer_dim)
        self.actout = nn.GELU(hidden_layer_dim)
        self.FFout = nn.Linear(hidden_layer_dim,output_dim)

    def block(self,A):

        A = self.FF1(A)
        values = self.value_act(A)
        values = self.valueFF(values)
        keys = self.key_act(A)
        keys = self.keyFF(keys)
        o = self.LQA(keys,values)
        o = self.FF2(o)
        o = self.actout(o)
        o = self.FFout(o)
        return o


#TODO write custom backward for this that allows for checkpointing:
# The default checkpointing won't work for some reason.

# we could return to using a normal tensor rather than a list (where we have to call stack every time, thus
# creating more memory overhead.
# By simply writing our own custom backward for the assignment to the memory tensor.
# Such a function would only need to store the part that's being assigned,
# Rather than storing the whole tensor as in stack. However, during the backprop step,
# the whole tensor would have to be reallocated. But trading speed for memory is the
# name of the game here.

class InnerForwardModel(nn.Module):
    def __init__(self, key_dim, embed_dim, core: nn.Module, key_head: nn.Module, value_head: nn.Module,
                 query_head: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 num_latents=512, n_external_heads=4):
        super(InnerForwardModel, self).__init__()

        self.external_mem_attn = MultiheadAttentionContainer(n_external_heads, tt.nn.InProjContainer(nn.Identity(), nn.Identity(),nn.Identity()),tt.nn.ScaledDotProduct(), nn.Identity(),batch_first = True)
        self.external_mem_w0 = nn.Linear(embed_dim, embed_dim)
        self.external_mem_ln = nn.LayerNorm(embed_dim)
        self.core = core
        self.key_head = key_head
        self.value_head = value_head
        self.query_head = query_head
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.key_dim = key_dim
    def forward(self,x, query,key_memory,value_memory, h,t,bsz):
        print(f"\tIter {t}")
        start = time.time()
        A = self.external_mem_attn(query, th.stack(key_memory,dim=1), th.stack(value_memory,dim=1))[0]
        A = th.squeeze(A)
        A = self.external_mem_w0(A)
        A = self.external_mem_ln(A)
        finish = time.time()
        print(f"\t\t External Memory Atttention Time: {finish - start}")
        A.view([bsz, self.num_latents, self.embed_dim])
        start = time.time()
        M = self.core(A,x,t=t)
        finish = time.time()
        print(f"\t\t Core Time: {finish - start}")
        start = time.time()
        key_memory.append(self.key_head(*M))
        value_memory.append(self.value_head(*M))
        h, query = self.query_head(*M, h=h)
        query = th.unsqueeze(query, 1)
        finish = time.time()
        print(f"\t\t Loop Head Time: {finish - start}")
        return query, key_memory, value_memory, h

class UniversalLearner(nn.Module):
    def __init__(self,key_dim,embed_dim, core : nn.Module, key_head : nn.Module, value_head : nn.Module, query_head :nn.Module,encoder : nn.Module, decoder : nn.Module,
                 num_latents = 512, n_external_heads = 4):
        super().__init__()
        self.inner_loop = InnerForwardModel(key_dim,embed_dim, core, key_head, value_head, query_head ,encoder , decoder,
                 num_latents, n_external_heads)
        self.encoder = encoder
        self.decoder = decoder
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.key_init = nn.Parameter(th.randn([num_latents, key_dim],requires_grad=True))
        self.value_init = nn.Parameter(th.randn([num_latents, embed_dim],requires_grad=True))
        self.query_init = nn.Parameter(th.randn([num_latents, key_dim],requires_grad=True))



    def forward(self,x,T = 10, with_checkpoint = False):

        start = time.time()
        bsz = x.size(0)
        key_memory = []
        value_memory = []
        key_memory.append(self.key_init.repeat([bsz,1]))
        value_memory.append(self.value_init.repeat([bsz,1]))
        query = th.unsqueeze(self.query_init.repeat([bsz,1]),1)
        x = self.encoder(x)
        finish = time.time()
        print(f"Initialization Time: {finish-start}")
        print(f"Begin iterations")
        h = None
        for t in range(T):
            print(f"\tIter {t}")
            if with_checkpoint:
                query, key_memory, value_memory, h = th.utils.checkpoint.checkpoint(self.inner_loop, *(x,query,key_memory,value_memory,h, t,bsz))
            else:
                query, key_memory, value_memory, h = self.inner_loop(x,query,key_memory,value_memory,h, t,bsz)

        start = time.time()
        key_memory = th.stack(key_memory,1)
        key_memory = key_memory.view(bsz,self.num_latents,T+1,self.key_dim)
        key_memory = key_memory[:,0,:,:]
        value_memory = th.stack(value_memory,1)
        value_memory = value_memory.view(bsz,self.num_latents,T+1,self.embed_dim)
        value_memory = value_memory[:,0,:,:]
        output = self.decoder(key_memory,value_memory)
        finish = time.time()
        print(f"Decoder Time: {finish - start}")

        return output

class ULfunc(th.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)

class ImageEncoder(nn.Module):
    def __init__(self,img_size,encoder_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(3,encoder_dim//2,7,stride=2,padding=3)
        self.act0 = nn.PReLU(encoder_dim//2)
        self.bn0 = nn.BatchNorm2d(encoder_dim//2)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(encoder_dim//2,encoder_dim//2,3,stride=1,padding=1)
        self.act1 = nn.PReLU(encoder_dim//2)
        self.bn1 = nn.BatchNorm2d(encoder_dim//2)

        self.key_conv = nn.Conv2d(encoder_dim//2,encoder_dim//2,3,stride=1,padding=1)
        self.key_act = nn.PReLU(encoder_dim//2)
        self.key_norm = nn.BatchNorm2d(encoder_dim//2)
        self.value_conv = nn.Conv2d(encoder_dim//2,encoder_dim//2,3,stride=1,padding=1)
        self.value_act = nn.PReLU(encoder_dim//2)
        self.value_norm = nn.BatchNorm2d(encoder_dim//2)

        self.value_PE = nn.Parameter(th.randn([encoder_dim//2,img_size//4, img_size//4],requires_grad=True))
        self.key_PE = nn.Parameter(th.randn([encoder_dim//2, img_size//4, img_size//4],requires_grad=True))


    def forward(self,x):
        x = self.conv0(x)
        x = self.act0(x)
        x = self.bn0(x)
        x = self.maxpool(x)
        h = self.conv1(x)
        h = self.act1(h)
        h = self.bn1(h)

        values = self.value_conv(h)
        values = self.value_act(values)
        values = self.value_norm(values)

        keys = self.key_conv(h)
        keys = self.key_act(keys)
        keys = self.key_norm(keys)

        values = th.cat([values, self.value_PE.expand([x.size(0),-1,-1,-1])],1)
        values = th.flatten(values,2,3)
        values = th.permute(values, [0,2,1])

        keys = th.cat([keys, self.key_PE.expand([x.size(0),-1,-1,-1])],1)
        keys = th.flatten(keys,2,3)
        keys = th.permute(keys, [0,2,1])
        return keys, values


class ImageEncoder(nn.Module):
    def __init__(self, img_size, encoder_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(3, encoder_dim // 2, 7, stride=2, padding=3)
        self.act0 = nn.PReLU(encoder_dim // 2)
        self.bn0 = nn.BatchNorm2d(encoder_dim // 2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(encoder_dim // 2, encoder_dim // 2, 3, stride=1, padding=1)
        self.act1 = nn.PReLU(encoder_dim // 2)
        self.bn1 = nn.BatchNorm2d(encoder_dim // 2)

        self.key_conv = nn.Conv2d(encoder_dim // 2, encoder_dim // 2, 3, stride=1, padding=1)
        self.key_act = nn.PReLU(encoder_dim // 2)
        self.key_norm = nn.BatchNorm2d(encoder_dim // 2)
        self.value_conv = nn.Conv2d(encoder_dim // 2, encoder_dim // 2, 3, stride=1, padding=1)
        self.value_act = nn.PReLU(encoder_dim // 2)
        self.value_norm = nn.BatchNorm2d(encoder_dim // 2)

        self.value_PE = nn.Parameter(th.randn([encoder_dim // 2, img_size // 4, img_size // 4], requires_grad=True))
        self.key_PE = nn.Parameter(th.randn([encoder_dim // 2, img_size // 4, img_size // 4], requires_grad=True))

    def forward(self, x):
        x = self.conv0(x)
        x = self.act0(x)
        x = self.bn0(x)
        x = self.maxpool(x)
        h = self.conv1(x)
        h = self.act1(h)
        h = self.bn1(h)

        values = self.value_conv(h)
        values = self.value_act(values)
        values = self.value_norm(values)

        keys = self.key_conv(h)
        keys = self.key_act(keys)
        keys = self.key_norm(keys)

        values = th.cat([values, self.value_PE.expand([x.size(0), -1, -1, -1])], 1)
        values = th.flatten(values, 2, 3)
        values = th.permute(values, [0, 2, 1])

        keys = th.cat([keys, self.key_PE.expand([x.size(0), -1, -1, -1])], 1)
        keys = th.flatten(keys, 2, 3)
        keys = th.permute(keys, [0, 2, 1])
        return keys, values


class SimpleEncoder(nn.Module):
    def __init__(self, img_size, encoder_dim, fourier_dim = None, num_bands = 4):
        super().__init__()
        if self.fourier_dim is None:
            self.fourier_dim = fourier_dim
        self.conv0 = nn.Conv2d(3, encoder_dim - fourier_dim, 7, stride=2, padding=3)
        self.act0 = nn.PReLU(encoder_dim - fourier_dim)
        self.bn0 = nn.BatchNorm2d(encoder_dim - fourier_dim)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(encoder_dim - fourier_dim, encoder_dim - fourier_dim, 3, stride=1, padding=1)
        self.act1 = nn.PReLU(encoder_dim - fourier_dim)
        self.bn1 = nn.BatchNorm2d(encoder_dim - fourier_dim)

        self.key_conv0 = nn.Conv2d(encoder_dim - fourier_dim, encoder_dim - fourier_dim, 3, stride=1, padding=)
        self.key_act = nn.PReLU(encoder_dim - fourier_dim)
        self.key_norm = nn.LayerNorm(encoder_dim - fourier_dim)
        self.key_conv1 = nn.Conv2d(encoder_dim, encoder_dim,1)

        self.value_conv0 = nn.Conv2d(encoder_dim - fourier_dim, encoder_dim - fourier_dim, 3, stride=1, padding=)
        self.value_act = nn.PReLU(encoder_dim - fourier_dim)
        self.value_norm = nn.LayerNorm(encoder_dim - fourier_dim)
        self.value_conv1 = nn.Conv2d(encoder_dim, encoder_dim,1)




    def forward(self, x):
        x = self.conv0(x)
        x = self.act0(x)
        x = self.maxpool(x)
        x = self.bn0(x)

        h = self.conv1(x)
        h = self.act1(h)
        h = self.bn1(h)

        values = self.value_conv0(h)
        values = self.value_act(values)
        values = self.value_norm(values)
        values = fourier_encode(values)
        values = self.value_conv1(values)

        keys = self.key_conv0(h)
        keys = self.key_act(keys)
        keys = self.key_norm(keys)
        keys = fourier_encode(keys)
        keys = self.key_conv1(keys)

        return keys, values


class TransformerCore(nn.Module):
    def __init__(self,external_key_dim, external_embed_dim,key_dim, embed_dim, num_latents, num_output_classes, img_size, encoder_dim=256, ff_dim= 2048, T=None):
        super().__init__()
        self.model = QIMIA_Sequential([InternalInitBlock(key_dim,embed_dim,external_embed_dim,num_latents,T=T),
                                FF_Block(key_dim,embed_dim,hidden_layer_dim=ff_dim),
                                SelfAttentionBlock(key_dim,embed_dim,num_latents),
                                # FF_Block(key_dim,embed_dim,hidden_layer_dim=ff_dim),
                                # SelfAttentionBlock(key_dim,embed_dim,num_latents),
                                CrossAttentionBlock(key_dim,embed_dim,encoder_dim,num_latents),
                                FF_Block(key_dim, embed_dim,hidden_layer_dim=ff_dim),
                                SelfAttentionBlock(key_dim, embed_dim, num_latents)]
                                # FF_Block(key_dim, embed_dim,hidden_layer_dim=ff_dim),
                                # SelfAttentionBlock(key_dim, embed_dim, num_latents),
                                # FF_Block(key_dim, embed_dim,hidden_layer_dim=ff_dim),
                                # SelfAttentionBlock(key_dim, embed_dim, num_latents)]
                                      )
    def forward(self,A,x,t=None):
        cross_keys,cross_values = x
        kwarg_list = []
        for block in self.model.blocks:
            if issubclass(block.__class__,InitializerBlock):
                kwarg_list.append({"t" : t})
            elif hasattr(block,'cross_attention'):
                kwarg_list.append({"cross_keys" :cross_keys, "cross_values":cross_values })
            else:
                kwarg_list.append({})
        return self.model(A,aux_list = kwarg_list)





def make_UL(external_key_dim, external_embed_dim,key_dim, embed_dim, num_latents, num_output_classes, img_size, encoder_dim=256, ff_dim= 2048,T = None):
        core =TransformerCore(external_key_dim, external_embed_dim,key_dim, embed_dim, num_latents, num_output_classes, img_size, encoder_dim=encoder_dim, ff_dim= 2048, T = T)
        num_params = sum(p.numel() for p in core.parameters() if p.requires_grad)
        print(f"number of core params: {num_params}")
        key_head  = HeadBlock(key_dim, embed_dim, external_key_dim)
        num_params = sum(p.numel() for p in key_head.parameters() if p.requires_grad)
        print(f"number of key_head params: {num_params}")
        value_head =  HeadBlock(key_dim, embed_dim, external_embed_dim)
        num_params = sum(p.numel() for p in value_head.parameters() if p.requires_grad)
        print(f"number of value_head params: {num_params}")
        query_head =  SimpleQueryHeadBlock(key_dim, embed_dim, external_key_dim)
        num_params = sum(p.numel() for p in query_head.parameters() if p.requires_grad)
        print(f"number of query_head params: {num_params}")
        encoder = ImageEncoder(img_size,encoder_dim)
        num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"number of image encoder params: {num_params}")
        decoder = HeadBlock(external_key_dim, external_embed_dim,num_output_classes)
        num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"number of Final Output Head params: {num_params}")
        return UniversalLearner(external_key_dim,external_embed_dim,core,key_head,value_head,query_head ,encoder, decoder, num_latents)

