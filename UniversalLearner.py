import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
import torchvision as tv
from FasterImplementation import BaseBlock, InitializerBlock, OutputBlock , TransformBlock, QIMIA_Sequential
from transformers import AutoTokenizer, PreTrainedTokenizer
from Vision import FF_Block
from utils import MultiheadAttentionContainer
import math



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
    def __init__(self,key_dim, embed_dim, external_embed_dim,num_latents):
        super().__init__()
        self.key_linear = nn.Linear(external_embed_dim,key_dim)
        self.value_linear = nn.Linear(external_embed_dim,embed_dim)
        self.key_PE = nn.Parameter(th.randn([num_latents,key_dim],requires_grad=True))
        self.value_PE = nn.Parameter(th.randn([num_latents, embed_dim],requires_grad=True))
        self.num_latents = num_latents
    def block(self,A):
        A = th.squeeze(A)
        key = self.key_linear(A)
        value = self.value_linear(A)
        key_PE = self.key_PE.repeat([A.size(0)//self.num_latents,1])
        value_PE = self.value_PE.repeat([A.size(0)//self.num_latents,1])
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
    def __init__(self,key_dim,embed_dim, output_dim, hidden_layer_dim = 2048):
        super().__init__(key_dim, embed_dim)
        self.ext_q_norm = nn.LayerNorm(key_dim)
        self.L1 = nn.Linear(embed_dim + key_dim, hidden_layer_dim)
        self.act1 = nn.GLU()
        self.L2 = nn.Linear(hidden_layer_dim//2,output_dim)
    def block(self,A , ext_q =None):
        o = th.cat([A,self.ext_q_norm(th.squeeze(ext_q))],dim=1)
        o = self.L1(o)
        o = self.act1(o)
        o = self.L2(o)
        return o
class UniversalLearner(nn.Module):
    def __init__(self,key_dim,embed_dim, core : nn.Module, key_head : nn.Module, value_head : nn.Module, query_head :nn.Module,encoder : nn.Module, decoder : nn.Module,
                 num_latents = 512):
        super().__init__()
        self.external_mem_attn = tt.nn.ScaledDotProduct(batch_first=True)
        self.external_mem_w0 = nn.Linear(embed_dim, embed_dim)
        self.external_mem_ln = nn.LayerNorm(embed_dim)
        self.core = core
        self.key_head = key_head
        self.value_head = value_head
        self.query_head = query_head
        self.encoder = encoder
        self.decoder = decoder
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.key_init = nn.Parameter(th.randn([num_latents, key_dim],requires_grad=True))
        self.value_init = nn.Parameter(th.randn([num_latents, embed_dim],requires_grad=True))
        self.query_init = nn.Parameter(th.randn([num_latents, key_dim],requires_grad=True))

    def forward(self,x,T = 10):
        bsz = x.size(0)
        key_memory = th.zeros([bsz*self.num_latents, T +1, self.key_dim]).to(x.device)
        value_memory = th.zeros([bsz*self.num_latents, T +1, self.embed_dim]).to(x.device)
        key_memory[:,0,:] = self.key_init.repeat([bsz,1])
        value_memory[:,0,:] = self.value_init.repeat([bsz,1])
        query = th.unsqueeze(self.query_init.repeat([bsz,1]),1)
        x = self.encoder(x)
        for t in range(T):
            A = self.external_mem_attn(query,key_memory[:,0:t,:].clone(),value_memory[:,0:t,:].clone())[0]
            A = self.external_mem_w0(A)
            A = self.external_mem_ln(A)
            A.view([bsz,self.num_latents,self.embed_dim])
            M = self.core(A,x)
            key_memory[:, t+1, :]  =key_memory[:, t+1, :]  + self.key_head(*M)
            value_memory[:, t+1, :] =  value_memory[:, t+1, :]+ self.value_head(*M)
            query = th.unsqueeze(self.query_head(*M,ext_q = query), 1)
        key_memory = key_memory.view(bsz,self.num_latents,T+1,self.key_dim)
        key_memory = key_memory[:,0,:,:]
        value_memory = value_memory.view(bsz,self.num_latents,T+1,self.key_dim)
        value_memory = value_memory[:,0,:,:]
        output = self.decoder(key_memory,value_memory)
        return output


class ImageEncoder(nn.Module):
    def __init__(self,img_size):
        super().__init__()
        self.conv0 = nn.Conv2d(3,64,7,stride=2,padding=3)
        self.act0 = nn.PReLU(64)
        self.bn0 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.act1 = nn.PReLU(64)
        self.bn1 = nn.BatchNorm2d(64)

        self.key_conv = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.key_act = nn.PReLU(64)
        self.key_norm = nn.BatchNorm2d(64)
        self.value_conv = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.value_act = nn.PReLU(64)
        self.value_norm = nn.BatchNorm2d(64)

        self.value_PE = nn.Parameter(th.zeros([64,img_size//4, img_size//4],requires_grad=True))
        self.key_PE = nn.Parameter(th.zeros([64, img_size//4, img_size//4],requires_grad=True))


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


class TransformerCore(nn.Module):
    def __init__(self,external_key_dim, external_embed_dim,key_dim, embed_dim, num_latents, num_output_classes, img_size, encoder_dim=256, ff_dim= 2048):
        super().__init__()
        self.model = QIMIA_Sequential([InternalInitBlock(key_dim,embed_dim,external_embed_dim,num_latents),
                                FF_Block(key_dim,embed_dim),
                                SelfAttentionBlock(key_dim,embed_dim,num_latents),
                                FF_Block(key_dim,embed_dim,external_embed_dim),
                                SelfAttentionBlock(key_dim,embed_dim,num_latents),
                                CrossAttentionBlock(key_dim,embed_dim,encoder_dim,num_latents),
                                FF_Block(key_dim, embed_dim),
                                SelfAttentionBlock(key_dim, embed_dim, num_latents),
                                FF_Block(key_dim, embed_dim, external_embed_dim),
                                SelfAttentionBlock(key_dim, embed_dim, num_latents),
                                FF_Block(key_dim, embed_dim, external_embed_dim),
                                SelfAttentionBlock(key_dim, embed_dim, num_latents)])
    def forward(self,A,x):
        cross_keys,cross_values = x
        kwarg_list = []
        for block in self.model.blocks:
            if hasattr(block,'cross_attention'):
                kwarg_list.append({"cross_keys" :cross_keys, "cross_values":cross_values })
            else:
                kwarg_list.append({})
        return self.model(A,aux_list = kwarg_list)





def make_UL(external_key_dim, external_embed_dim,key_dim, embed_dim, num_latents, num_output_classes, img_size, encoder_dim=256, ff_dim= 2048):
        core =TransformerCore(external_key_dim, external_embed_dim,key_dim, embed_dim, num_latents, num_output_classes, img_size, encoder_dim=encoder_dim, ff_dim= 2048)
        key_head  = HeadBlock(key_dim, embed_dim, external_key_dim)
        value_head =  HeadBlock(key_dim, embed_dim, external_embed_dim)
        query_head =  QueryHeadBlock(key_dim, embed_dim, external_key_dim)
        encoder = ImageEncoder(img_size)
        decoder = HeadBlock(external_key_dim, external_embed_dim,num_output_classes)
        return UniversalLearner(external_key_dim,external_embed_dim,core,key_head,value_head,query_head ,encoder, decoder, num_latents)

