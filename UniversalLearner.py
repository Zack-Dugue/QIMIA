import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from Implementation import BaseBlock, InitializerBlock, OutputBlock , TransformBlock, QIMIA_Sequential
from transformers import AutoTokenizer, PreTrainedTokenizer
from NLP import FF_Block, AttentionBlock, GeGLU
import math
F.multi_head_attention_forward()

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

class MemoryBlock(nn.Module):
    def __init__(self, external_key_dim, external_embed_dim, internal_key_dim, internal_embed_dim, num_external_heads = 8, hidden_dim = 512):
        super().__init__()
        self.external_memory_attention = tt.nn.ScaledDotProduct()
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
        return A

class HeadBlock(OutputBlock):
    def __init__(self,key_dim,embed_dim, output_dim, hidden_layer_dim = 2048):
        super().__init__(key_dim, embed_dim)
        self.L1 = nn.Linear(embed_dim, hidden_layer_dim)
        self.act1 = GeGLU(hidden_layer_dim)
        self.L2 = nn.Linear(hidden_layer_dim//2,output_dim)

class UniversalLearner(nn.Module):
    def __init__(self,key_dim,embed_dim, core : nn.Module, key_head : nn.Module, value_head : nn.Module, query_head :nn.Module,encoder : nn.Module, decoder : nn.Module,
                 num_latents = 512):
        super().__init__()
        self.external_mem_attn = MemoryBlock(external_key_dim,external_embed_dim,internal_key_dim,internal_embed_dim)



