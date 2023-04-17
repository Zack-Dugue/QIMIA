import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from container import QIMIA , QIMIA2, LearnedQueryAttention , LearnedQueryAttention2
from transformers import AutoTokenizer, PreTrainedTokenizer
import math
class GeGELU(nn.Module):
    def __init__(self,width):
        super(GeGELU, self).__init__()
        self.act = nn.GELU()
        assert(width % 2 == 0)
        self.width = width
    def forward(self,x):
        out = x[:,:,:self.width//2] * self.act(x[:,:,self.width//2:])
        return out


class AttentionBlock(nn.Module):
    def __init__(self,embed_dim,n_heads, w0=True,):
        super(AttentionBlock, self).__init__()
        self.key_proj =  nn.Linear(embed_dim,embed_dim)
        self.query_proj = nn.Linear(embed_dim,embed_dim)
        self.value_project = nn.Linear(embed_dim,embed_dim)
        if w0:
            self.w0 = nn.Linear(embed_dim,embed_dim)
        else:
            self.w0 = nn.Identity()
        self.attention =nn.MultiheadAttention(embed_dim,n_heads)
    def forward(self,x):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_project(x)
        o = self.attention(Q,K,V)[0]
        return self.w0(o)

class FF_Layer(nn.Module):
    def __init__(self, embed_dim, hidden_dim = 1024):
        super(FF_Layer, self).__init__()

        self.ff1 = nn.Linear(embed_dim, hidden_dim)
        self.act1 = GeGELU(hidden_dim)
        self.ff2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.act2 = GeGELU(hidden_dim)
        # self.ff3 = nn.Linear(hidden_dim // 2, embed_dim)
    def forward(self,x):
        x = self.ff1(x)
        x = self.act1(x)
        x = self.ff2(x)
        x = self.act2(x)
        # x = self.ff3(x)
        return x

#Stolen from pytorch wiki
def make_positional_encoding(d_model: int, max_len: int = 5000):

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = th.sin(position * div_term)
        pe[:, 0, 1::2] = th.cos(position * div_term)
        return pe


def make_nlp_QIMIA(init_dim, embed_dim,key_dim, num_layers,hidden_dim = 1024):
    """

    :param init_dim: the initial dimension of the input
    :param embed_dim: the embedding dimension of the model
    :param key_dim: the desired key dimension for KIMIA
    :param num_layers: the number of layers
    :return:
    """
    blocks = []
    key_project = []
    value_project = []
    memory_attention = []
    input_key_project = nn.Linear(init_dim,key_dim)
    input_value_project = nn.Linear(init_dim,embed_dim)

    for i in range(num_layers):
        blocks.append(AttentionBlock(embed_dim,8))
        blocks.append(FF_Layer(embed_dim))
        key_project.append(nn.Sequential(nn.PReLU(embed_dim),nn.Linear(embed_dim,key_dim)))
        key_project.append(nn.Linear(hidden_dim//2,key_dim))
        value_project.append(nn.Sequential(nn.PReLU(embed_dim),nn.Linear(embed_dim,embed_dim)))
        value_project.append(nn.Linear(hidden_dim//2,embed_dim))

        mia1 = nn.Sequential(LearnedQueryAttention(key_dim,8,v_dim=embed_dim, w0= True),nn.LayerNorm(embed_dim))
        mia2 = nn.Sequential(LearnedQueryAttention(key_dim,8,v_dim=embed_dim, w0= True),nn.LayerNorm(embed_dim))
        memory_attention.append(mia1)
        memory_attention.append(mia2)

    return QIMIA(blocks,key_project,value_project,input_key_project,input_value_project,memory_attention)


def make_nlp_QIMIA2(embed_dim,key_dim, num_layers,hidden_dim = 1024):
    """

    :param init_dim: the initial dimension of the input
    :param embed_dim: the embedding dimension of the model
    :param key_dim: the desired key dimension for KIMIA
    :param num_layers: the number of layers
    :return:
    """
    blocks = []
    key_project = []
    value_project = []
    memory_attention = []

    for i in range(num_layers):
        blocks.append(AttentionBlock(embed_dim,8))
        blocks.append(FF_Layer(embed_dim))
        key_project.append(nn.Sequential(nn.PReLU(embed_dim),nn.Linear(embed_dim,key_dim)))
        key_project.append(nn.Linear(hidden_dim//2,key_dim))
        value_project.append(nn.Sequential(nn.PReLU(embed_dim),nn.Linear(embed_dim,embed_dim)))
        value_project.append(nn.Linear(hidden_dim//2,embed_dim))

        mia1 = LearnedQueryAttention(key_dim,8,v_dim=embed_dim, w0= True)
        mia2 = LearnedQueryAttention(key_dim,8,v_dim=embed_dim, w0= True)
        memory_attention.append(mia1)
        memory_attention.append(mia2)

    return QIMIA2(blocks,key_project,value_project,memory_attention)
