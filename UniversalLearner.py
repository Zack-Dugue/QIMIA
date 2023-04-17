import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from Implementation import BaseBlock, InitializerBlock, OutputBlock , TransformBlock, QIMIA_Sequential
from transformers import AutoTokenizer, PreTrainedTokenizer
from NLP import FF_Block, AttentionBlock, GeGLU
import math

class CrossAttentionBlock(BaseBlock):
    def __init__(self,key_dim, embed_dim, cross_dim, num_cross_heads=1):
        # Assumes the value dimension and key dimension of the encoder
        # are the same.
        super().__init__(key_dim,embed_dim)
        self.Wq = nn.Linear(embed_dim,cross_dim)
        self.cross_attention == nn.MultiheadAttention(cross_dim,num_cross_heads)
        self.value_act = nn.PReLU(cross_dim)
        self.value_project = nn.Linear(cross_dim,embed_dim)
        self.key_act = nn.PReLU(cross_dim)
        self.key_project = nn.Linear(cross_dim,key_dim)

    def block(self,A, num_latents=None, cross_keys = None, cross_values = None):
        if num_latents == None:
            raise ValueError("Num Latents must be defined for Cross Atention")
        if cross_keys == None:
            raise ValueError("The keys for cross attention must be defined")
        if cross_values == None:
            raise ValueError("The values for cross attention must be defined")

        A = A.view(A.size(0)//num_latents, num_latents , A.size(1))
        Q = self.Wq(A)
        o = self.cross_attention(Q,cross_keys,cross_values)[0]
        key = self.key_act(o)
        key = self.key_project(key)

        value = self.value_act(o)
        value = self.value_project(value)

        return key, value

class MemoryBlock(InitializerBlock):
    def __init__(self, external_key_dim, external_embed_dim, internal_key_dim, internal_embed_dim, num_external_heads = 8, hidden_dim = 512):
        super().__init__()
        self.external_memory_attention = nn.MultiheadAttention(external_embed_dim, num_external_heads, kdim= external_key_dim, batch_first=True)
        self.norm0 = nn.LayerNorm(external_embed_dim)
        self.linear0 = nn.Linear(external_embed_dim,hidden_dim)
        #these PReLU's probably won't work.
        self.key_act = nn.PReLU(hidden_dim)
        self.key_project = nn.Linear(hidden_dim,internal_key_dim)
        self.value_act = nn.PReLU(hidden_dim)
        self.value_project = nn.Linear(hidden_dim, internal_embed_dim)
    def block(self,keys, values, query):
        # should be of the proper dimensionality.
        A = self.external_memory_attention(keys,values,query)
        A = self.norm0(A)
        A = self.linear0(A)
        key = self.key_act(A)
        key = self.key_project(key)
        value = self.value_act(A)
        value = self.value_project(value)
        return key , value

class HeadBlock(OutputBlock):
    def __init__(self,key_dim,embed_dim, output_dim, hidden_layer_dim = 2048):
        super().__init__(key_dim, embed_dim)
        self.L1 = nn.Linear(embed_dim, hidden_layer_dim)
        self.act1 = GeGLU(hidden_layer_dim)
        self.L2 = nn.Linear(hidden_layer_dim//2,output_dim)

class UniversalLearner(nn.Module):
    def __init__(self, external_key_dim, external_embed_dim,  internal_key_dim, internal_embed_dim, core,encoder, decoder,
                 num_latents = 512):
        super().__init__()
        self.external_mem_attn = MemoryBlock(external_key_dim,external_embed_dim,internal_key_dim,internal_embed_dim)
        self.query_head  = HeadBlock(internal_key_dim, internal_embed_dim, external_key_dim)
        self.key_head  = HeadBlock(internal_key_dim, internal_embed_dim, external_key_dim)
        self.value_head = HeadBlock(internal_key_dim, internal_embed_dim, external_embed_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.core = core
        self.num_latents = num_latents
        # todo:
        #  handle the initial external values, and external keys and external queries
        #  just seems like a massive pain tbh ripppp.
    def forward(self,x,num_iterations = 10):
        # for i in range(num_iterations-1):


