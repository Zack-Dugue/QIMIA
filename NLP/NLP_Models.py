import torchtext as tt
import torch as th
import torch.nn as nn
from Implementation import BaseBlock, InitializerBlock, OutputBlock, TransformBlock, QIMIA_Sequential
from utils import P_SIGLU
import torch.nn.functional as F
import math
class Test_FF_Block(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256):
        super(Test_FF_Block, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.L0 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.L1 = nn.Linear(hidden_dim, embed_dim)

    def forward(self,x):
        x = self.norm(x)
        x = self.L0(x)
        x = self.act(x)
        x = self.L1(x)
        return x



class Test_Attention_Block(nn.Module):
    def __init__(self,embed_dim, num_heads, causal = False):
        super(Test_Attention_Block, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.causal = causal

    def set_causal(self, causal : bool):
        self.causal = causal

    def forward(self,x):
        x = self.norm(x)
        x = self.attention(x,x,x,is_causal=self.causal)[0]


        return x


class FFBlock(BaseBlock):
    def __init__(self, key_dim, embed_dim, hidden_dim=256):
        super(FFBlock, self).__init__(key_dim,embed_dim)
        self.L0 = nn.Linear(embed_dim, hidden_dim)
        self.key_act = P_SIGLU(hidden_dim)
        self.Lkey = nn.Linear(hidden_dim, key_dim)
        self.value_act = P_SIGLU(hidden_dim)
        self.Lvalue = nn.Linear(hidden_dim, embed_dim)

    def block(self,x):
        x = self.L0(x)
        key = self.key_act(x)
        key = self.Lkey(key)
        value = self.value_act(x)
        value = self.Lvalue(value)
        return key , value

class AttentionBlock(BaseBlock):
    def __init__(self,key_dim, embed_dim, num_heads, causal = False):
        super(AttentionBlock, self).__init__(key_dim,embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.key_act = P_SIGLU(embed_dim)
        self.Lkey = nn.Linear(embed_dim,key_dim)
        self.value_act = P_SIGLU(embed_dim)
        self.Lvalue = nn.Linear(embed_dim,embed_dim)

    def set_causal(self,causal : bool):
        self.causal = causal

    def block(self,A, num_tokens = None):
        A = A.view([-1,num_tokens,A.size(-1)])
        A = self.attention(A,A,A,is_causal=self.causal)[0]
        key = self.key_act(A)
        key = self.Lkey(key)
        value = self.value_act(A)
        value = self.Lvalue(value)

        return th.flatten(key,0,1) , th.flatten(value,0,1)

#stolen from pytorch website
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(1, max_len, d_model)
        pe[0,:, 0::2] = th.sin(position * div_term)
        pe[0,:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: th.Tensor, residual = True) -> th.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if residual:
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
        else:
            return self.dropout(self.pe[:, :x.size(1), :])

class NLP_Initializer(InitializerBlock):
    def __init__(self,key_dim,embed_dim,vocab_size,max_len = 5000):
        super().__init__()
        self.ValueEmbedding = nn.Embedding(vocab_size,embed_dim)
        self.KeyEmbedding = nn.Embedding(vocab_size,key_dim)
        self.PE = PositionalEncoding(embed_dim)
        self.PE_Key = nn.Linear(embed_dim,key_dim)
        self.PE_Value = nn.Linear(embed_dim,embed_dim)

    def block(self,x):
        embed_value = self.ValueEmbedding(x)
        embed_key = self.KeyEmbedding(x)
        pe_key = self.PE_Key(self.PE(x,residual=False)).repeat([x.size(0),1,1])
        pe_value = self.PE_Value(self.PE(x,residual=False))
        return [th.flatten(embed_key,0,1), th.flatten(pe_key,0,1)] , [th.flatten(embed_value,0,1), th.flatten(pe_value,0,1)]

class LM_Head(OutputBlock):
    def __init__(self,key_dim,embed_dim,vocab_size,hidden_dim=1024):
        super(LM_Head, self).__init__(key_dim,embed_dim)
        self.L0 = nn.Linear(embed_dim,hidden_dim)
        self.act = nn.GELU()
        self.L1 = nn.Linear(hidden_dim,vocab_size)
    def block(self,A):
        A = self.L0(A)
        A = self.act(A)
        logits = self.L1(A)
        return logits

class QIMIA_Transformer(nn.Module):
    def __init__(self,key_dim,embed_dim, input_token_dim, num_output_classes, num_layers, input_attention_heads = 8, FF_hidden_dim=1024,output_hidden_dim = 2048):
        super().__init__()
        blocks = [NLP_Initializer(key_dim,embed_dim,input_token_dim)]
        for i in range(num_layers):
            blocks.append(AttentionBlock(key_dim, embed_dim, input_attention_heads))
            blocks.append(FFBlock(key_dim,embed_dim,hidden_dim=FF_hidden_dim))
        blocks.append(LM_Head(key_dim,embed_dim,num_output_classes))
        # blocks[1].LQA.diagnostic = True
        self.model = QIMIA_Sequential(blocks)


    def parameters(self):
        params = self.model.parameters()
        return self.model.parameters()


    def forward(self,x):
        kwarg_list = []
        for i in range(len(self.model)-1):
            if i % 2 == 1:
                kwarg_list.append({"num_tokens" : x.size(1)})
            else:
                kwarg_list.append({})
        kwarg_list.append({})
        return self.model(x,aux_list = kwarg_list)


# class QIMIA_Transformer(nn.Module):

class Test_Transformer(nn.Module):
    def __init__(self,num_layers,vocab_size,embed_dim,n_heads, causal = False):
        super(Test_Transformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(Test_FF_Block(embed_dim,hidden_dim = 256))
            self.layers.append(Test_Attention_Block(embed_dim,n_heads,causal = causal))
        self.head = nn.Linear(embed_dim,vocab_size)
        self.causal = True

    def forward(self,x,causal = True):
        x = self.pe(self.embed(x))
        for layer in self.layers:
            x = x + layer(x)
        x = self.head(x)
        return x

def make_causal_mask(x):
  mask = th.triu(th.ones([x.size(1),x.size(1)]),diagonal=1).bool()
  mask = mask.repeat([x.size(0),1,1])
  return mask

if __name__ == "__main__":
    x = th.ones([6,10,5])
    mask  = make_causal_mask(x)
    print(f"x is - {x}")
    print(f"mask is - {mask}")

