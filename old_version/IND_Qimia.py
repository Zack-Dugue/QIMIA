import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from container import QIMIA , QIMIA2, LearnedQueryAttention , LearnedQueryAttention2
from transformers import AutoTokenizer, PreTrainedTokenizer
from NLP import FF_Layer, GeGELU
import math

class FFChunk(nn.Module):
    def __init__(self,key_dim, embed_dim,hidden_dim, num_lqa_heads = 8):
        super(FFChunk,self).__init__()

        self.LQA = LearnedQueryAttention(key_dim,num_lqa_heads,v_dim=embed_dim,w0=True)
        self.FF1 = nn.Linear(embed_dim,hidden_dim)
        self.act1 = GeGELU(hidden_dim)
        self.FF2 = nn.Linear(hidden_dim//2,hidden_dim)
        self.act2 =  GeGELU(hidden_dim)
        self.FFKey = nn.Linear(hidden_dim//2,key_dim)
        self.FFValue = nn.Linear(hidden_dim//2,embed_dim)
    def forward(self,keys,values):
        x = self.LQA(keys,values)
        x = self.FF1(x)
        x = self.act1(x)
        x = self.FF2(x)
        x = self.act2(x)
        key = self.FFKey(x)
        keys = th.cat([keys,th.unsqueeze(key,1)],1)
        value = self.FFValue(x)
        values = th.cat([value,th.unsqueeze(value,1)],1)
        return keys , values

class SelfAttentionChunk(nn.Module):
    def __init__(self,key_dim, embed_dim,num_attention_heads=8, num_lqa_heads = 8):
        super(SelfAttentionChunk,self).__init__()

        self.LQA = LearnedQueryAttention(key_dim,num_lqa_heads,v_dim=embed_dim,w0=True)
        self.KeyProj = nn.Linear(embed_dim,embed_dim)
        self.QueryProj = nn.Linear(embed_dim,embed_dim)
        self.ValueProj = nn.Linear(embed_dim,embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_attention_heads,batch_first=True)
        self.ActIntegratedKey = nn.PReLU(embed_dim)
        self.ActIntegratedValue = nn.PReLU(embed_dim)
        self.FFIntegratedKey = nn.Linear(embed_dim,key_dim)
        self.FFIntegratedValue = nn.Linear(embed_dim,embed_dim)

    def forward(self,keys,values):
        x = self.LQA(keys,values)
        K = self.KeyProj(x)
        V = self.ValueProj(x)
        Q = self.QueryProj(x)
        x = self.attention(K,Q,V)
        key = self.ActIntegratedKey(x)
        key = self.FFIntegratedKey(key)
        value = self.ActIntegratedValue(x)
        value = self.FFIntegratedValue(value)
        keys = th.cat([keys,th.unsqueeze(key,1)],1)
        values = th.cat([value,th.unsqueeze(value,1)],1)
        return keys , values

class CrossAttentionChunk(nn.Module):
    # More can be done here to handle the case where the channel dimensionlaity of the encoded
    # Input is different from the Latent channel Dimensionality.
    def __init__(self,key_dim, embed_dim, num_ca_heads, num_lqa_heads = 8):

        """

        :param key_dim:
        :param embed_dim:
        :param num_ca_heads:
        :param num_lqa_heads:
        """
        super(CrossAttentionChunk,self).__init__()

        self.LQA = LearnedQueryAttention(key_dim,num_lqa_heads,v_dim=embed_dim,w0=True)
        #This Query Project might be redundant... but this is important enough to waste the parameters on.
        # And Layer Norm really can act like a nonlinearity at times.
        self.QueryProject = nn.Linear(embed_dim,embed_dim)
        self.CrossAttention = nn.MultiheadAttention(embed_dim,batch_first=True)

        self.ActIntegratedKey = nn.PReLU(embed_dim)
        self.ActIntegratedValue = nn.PReLU(embed_dim)
        self.FFIntegratedKey = nn.Linear(embed_dim,key_dim)
        self.FFIntegratedValue = nn.Linear(embed_dim,embed_dim)
    def forward(self,keys,values,K,V):
        """

        :param keys: keys from QIMIA
        :param values: values from QIMIA
        :param K: keys from encoded input
        :param V: values from encoded input
        :return:
        """
        Q = self.LQA(keys,values)
        Q = self.QueryProject(Q)
        A = self.attention(Q,K,V)
        key = self.actKey(A)
        key = self.FFKey(key)
        value = self.actValue(A)
        value = self.FFValue(value)
        keys = th.cat([keys,th.unsqueeze(key,1)],1)
        values = th.cat([value,th.unsqueeze(value,1)],1)
        return keys , values

class OutPutHead(nn.Module):
    # More can be done here to handle the case where the channel dimensionlaity of the encoded
    # Input is different from the Latent channel Dimensionality.
    def __init__(self,key_dim, embed_dim,output_dim, hidden_dim=2048, num_lqa_heads = 8):

        """

        :param key_dim:
        :param embed_dim:
        :param num_ca_heads:
        :param num_lqa_heads:
        """
        super(OutPutHead,self).__init__()

        self.LQA = LearnedQueryAttention(key_dim,num_lqa_heads,v_dim=embed_dim,w0=True)
        self.FF1 = nn.Linear(embed_dim,hidden_dim)
        self.act1 = GeGELU(hidden_dim)
        self.FF2 = nn.Linear(hidden_dim//2,output_dim)
    def forward(self,keys,values):
        A = self.LQA(keys,values)
        A = self.FF1(A)
        A = self.act1(A)
        A = self.FF2(A)
        return A

class EntranceChunk(nn.Module):
    def __init__(self, internal_key_dim, internal_embed_dim , external_embed_dim, hidden_dim=2048, num_external_heads=8):
        """

        :param key_dim:
        :param embed_dim:
        :param num_ca_heads:
        :param num_lqa_heads:
        """
        super(EntranceChunk, self).__init__()
        self.ExternalAttention = nn.MultiheadAttention(external_embed_dim,num_external_heads)
        self.norm = nn.LayerNorm(external_embed_dim)
        self.FFValue = nn.Linear(external_embed_dim,internal_embed_dim)
        self.FFKey = nn.Linear(external_embed_dim,internal_key_dim)
    def forward(self,external_keys, external_values, query):
        # needs to include info about the number of latents
        # because the latent dim of the External Tokens will be flattened along batch dimension.
        E = self.ExternalAttention(external_keys,external_values,query)
        E = self.norm(E)
        Values = th.unsqueeze(self.FFValue(E),1)
        # Values = th.

#For now we are assuming that the dimensionality of the internal QIMIA
# is the same as the dimensionality of the External QIMIA
# but this can be chagned in the future without too much trouble

class UniversalLearner(nn.Module):
    def __init__(self, block, num_latents, embed_dim,key_dim, encoder,decoder, LQA_attention_heads = 8,self_attention_heads=16,
                 cross_attention_heads = 4, FF_hidden_dim = 1024 ):
        super(UniversalLearner, self).__init__()
        """
        block - a list of characters S, F, C
                where A is a self attention layer
                where F is a feed foward Layer
                and where C is a cross attention layer to the input
        """
        core = nn.ModuleList()
        for char in block:
            if char not in ['S','F', 'C']:
                raise ValueError
            if char == 'S':
                core.append(SelfAttentionChunk(key_dim,embed_dim,num_lqa_heads=LQA_attention_heads))
            if char == 'F':
                core.append(FFChunk(key_dim,embed_dim,hidden_dim=FF_hidden_dim,num_lqa_heads=LQA_attention_heads))
            if char == 'C':
                core.append(CrossAttentionChunk(key_dim,embed_dim,self_attention_heads,num_lqa_heads=LQA_attention_heads))
        self.ExternalLQA =
        self.core = core
        self.key_head = OutPutHead(key_dim,embed_dim,key_dim)
        self.query_head = OutPutHead(key_dim,embed_dim,key_dim)
        self.value_head = OutPutHead(key_dim,embed_dim,embed_dim)

        self.decoder = decoder
        self.init_latent_value = th.randn([num_latents,embed_dim],requires_grad = True)
        self.init_latent_key = th.randn([num_latents,embed_dim], requires_grad = True)
        self.embed_dim = embed_dim
        self.key_dim = key_dim
    def forward(self,x,num_iterations = 10):
        bsz = x.size(0)
        input_keys , input_values = self.encode(x)
        keys = th.unsqueeze(self.init_latent_key,1).expand([bsz,1,self.key_dim])
        values = th.unsqueeze(self.init_latent_value,1).expand([bsz,1,self.embed_dim])
        for iteration in range(num_iterations):
            for chunk in self.core:
                if chunk.type ==CrossAttentionChunk:
                    keys, values = chunk(keys,values,input_keys,input_values)
                else:
                    keys, values = chunk(keys,values)

        return ou

