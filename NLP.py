import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from Implementation import BaseBlock, InitializerBlock, OutputBlock , TransformBlock, QIMIA_Sequential
from transformers import AutoTokenizer, PreTrainedTokenizer
import math

class PReLU(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.actual_prelu = nn.PReLU(embed_dim)
    def forward(self,x):
        x = th.permute(x,[0,2,1])
        x = self.actual_prelu(x)
        x = th.permute(x,[0,2,1])
        return x
class GeGLU(nn.Module):
    def __init__(self,width):
        super(GeGLU, self).__init__()
        self.act = nn.GELU()
        assert(width % 2 == 0)
        self.width = width
    def forward(self,x):
        out = x[:,:self.width//2] * self.act(x[:,self.width//2:])
        return out

class FF_Block(BaseBlock):
    def __init__(self,key_dim,embed_dim, hidden_layer_dim = 1024):
        super().__init__(key_dim,embed_dim)
        self.L1 = nn.Linear(embed_dim,hidden_layer_dim)
        self.act1 = GeGLU(hidden_layer_dim)
        self.L2 = nn.Linear(hidden_layer_dim//2,hidden_layer_dim)
        self.act2 = GeGLU(hidden_layer_dim)
        self.KeyL = nn.Linear(hidden_layer_dim//2,key_dim)
        self.ValueL = nn.Linear(hidden_layer_dim//2,embed_dim)
    def block(self, A):
        A = self.L1(A)
        A = self.act1(A)
        A = self.L2(A)
        A = self.act2(A)
        key = self.KeyL(A)
        value = self.ValueL(A)
        return key, value

class AttentionBlock(BaseBlock):
    def __init__(self,key_dim,embed_dim, num_attention_heads):
        super().__init__(key_dim,embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_attention_heads,batch_first=True)
        self.key_act = PReLU(embed_dim)
        self.linear_key_out = nn.Linear(embed_dim,key_dim)
        self.value_act = PReLU(embed_dim)
        self.linear_value_out = nn.Linear(embed_dim,embed_dim)
    def block(self,A,num_input_tokens= None):
        if num_input_tokens == None:
            raise ValueError("Need to specify the numebr of tokens in the input to" \
                             "an attention layer for NLP QIMIA")
        A = A.view(A.size(0)//num_input_tokens, num_input_tokens, A.size(1))
        Q = self.Wq(A)
        K = self.Wk(A)
        V = self.Wv(A)
        o = self.attention(Q,K,V)[0]
        key = self.key_act(o)
        key = self.linear_key_out(key)
        value = self.value_act(o)
        value = self.linear_value_out(value)
        return key , value

class OutputClassifierBlock(OutputBlock):
    def __init__(self,key_dim,embed_dim,output_dim, hidden_dim = 2048,diagnostic=False):
        super().__init__(key_dim,embed_dim,diagnostic=diagnostic)
        self.pre_pool_linear = nn.Linear(embed_dim,embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.FF1 = nn.Linear(embed_dim, hidden_dim)
        self.Act1 = GeGLU(hidden_dim)
        self.FF2 = nn.Linear(hidden_dim//2, hidden_dim)
        self.Act2 = GeGLU(hidden_dim)
        self.FF3 = nn.Linear(hidden_dim//2,output_dim)
    def block(self, A,num_input_tokens=None):
        if num_input_tokens == None:
            raise ValueError("Need to specify the numebr of tokens in the input to " \
                             "an output layer for NLP QIMIA")
        A = A.view(A.size(0)//num_input_tokens, num_input_tokens, A.size(2))
        A = th.squeeze(F.avg_pool1d(th.permute(A,[0,2,1]), kernel_size=num_input_tokens, stride=1))

        A = self.output_norm(A)
        A = self.FF1(A)
        A = self.Act1(A)
        A = self.FF2(A)
        A = self.Act2(A)
        A = self.FF3(A)
        return A

class NLP_EncoderBlock(InitializerBlock):
    def __init__(self,key_dim,embed_dim,input_token_dim,max_input_width = 10000):
        super().__init__()
        self.input_key_embed = nn.Linear(input_token_dim, key_dim)
        self.input_val_embed = nn.Linear(input_token_dim, embed_dim)
        self.PE_key_encode = nn.Linear(embed_dim,embed_dim)
        self.PE_val_encode = nn.Linear(embed_dim,embed_dim)
        self.PE =  self.make_positional_encoding(embed_dim,max_len=max_input_width)
        #implement token flattening thing a ma doohicky
        self.flatten_to_token = nn.Flatten(0,1)

    @staticmethod
    #stull this from some towards data science article
    def make_positional_encoding(d_model: int, max_len: int = 5000):

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = th.sin(position * div_term)
        pe[:, 0, 1::2] = th.cos(position * div_term)
        return pe
    def block(self, x):
        x_key = self.input_key_embed(self.flatten_to_token(x))
        x_val = self.input_val_embed(self.flatten_to_token(x))
        PE_key = self.flatten_to_token(self.PE_key_encode(self.PE[:x.size(1)]).repeat([x.size(0),1,1]))
        PE_val = self.flatten_to_token(self.PE_val_encode(self.PE[:x.size(1)]).repeat([x.size(0),1,1]))
        # return  [x_key,PE_key] , [x_val, PE_val]
        return x_key , x_val

class NLPClassifier(nn.Module):
    def __init__(self,key_dim,embed_dim, input_token_dim, num_output_classes, num_layers, input_attention_heads = 8, FF_hidden_dim=1024,output_hidden_dim = 2048):
        super().__init__()
        blocks = [NLP_EncoderBlock(key_dim,embed_dim,input_token_dim)]
        for i in range(num_layers):
            blocks.append(AttentionBlock(key_dim, embed_dim, input_attention_heads))
            blocks.append(FF_Block(key_dim,embed_dim,hidden_layer_dim=FF_hidden_dim))
        blocks.append(OutputClassifierBlock(key_dim,embed_dim,num_output_classes,diagnostic=False))
        # blocks[1].LQA.diagnostic = True
        self.model = QIMIA_Sequential(blocks)


    def parameters(self):
        params = self.model.parameters()
        return self.model.parameters()
    def forward(self,x):
        kwarg_list = []
        for i in range(len(self.model)):
            if i % 2 == 1:
                kwarg_list.append({"num_input_tokens" : x.size(1)})
            else:
                kwarg_list.append({})
        return self.model(x,aux_list = kwarg_list)

class FF_Encoder_Layer_Component(nn.Module):
    def __init__(self,embed_dim, hidden_layer_dim = 1024):
        super().__init__()
        self.L1 = nn.Linear(embed_dim,hidden_layer_dim)
        self.act1 = GeGLU(hidden_layer_dim)
        self.L2 = nn.Linear(hidden_layer_dim//2,embed_dim)

    def forward(self, A):
        A = self.L1(A)
        A = self.act1(A)
        A = self.L2(A)
        return A

class AttentionEncoderLayer(nn.Module):
    def __init__(self,embed_dim, num_attention_heads):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_attention_heads,batch_first=True)

    def forward(self,A):
        Q = self.Wq(A)
        K = self.Wk(A)
        V = self.Wv(A)
        o = self.attention(Q,K,V)[0]
        return o
class EncoderLayer(BaseBlock):
    def __init__(self, key_dim, embed_dim, num_heads):
        super().__init__(embed_dim,key_dim,)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.AttentionLayer = AttentionEncoderLayer(embed_dim,16)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.FF_Layer = FF_Encoder_Layer_Component(embed_dim)
        self.ValueOut = nn.Linear(embed_dim,embed_dim)
        self.KeyOut = nn.Linear(embed_dim,key_dim)


    def block(self,A,num_input_tokens = None):
        A = A.view([A.size(0)//num_input_tokens, num_input_tokens, A.size(1)])
        A = A +  self.AttentionLayer(self.norm1(A))
        A = th.flatten(A,0,1)
        A = A +  self.FF_Layer(self.norm2(A))
        key = self.KeyOut(A)
        value = self.ValueOut(A)
        return key, value

#This one uses a through block encoder layer instead of seperate FF_blocks or attention blocks.
class NLPClassifier2(nn.Module):
    def __init__(self,key_dim,embed_dim, input_token_dim, num_output_classes, num_layers, input_attention_heads = 8, FF_hidden_dim=1024,output_hidden_dim = 2048):
        super().__init__()
        blocks = [NLP_EncoderBlock(key_dim,embed_dim,input_token_dim)]
        for i in range(num_layers):
            blocks.append(EncoderLayer(key_dim,embed_dim, input_attention_heads))
        blocks.append(OutputClassifierBlock(key_dim,embed_dim,num_output_classes,diagnostic=False))
        self.model = QIMIA_Sequential(blocks)


    def parameters(self):
        params = self.model.parameters()
        return self.model.parameters()
    def forward(self,x):
        kwarg_list = []
        for i in range(len(self.model)):
            if i != 0 :
                kwarg_list.append({"num_input_tokens" : x.size(1)})
            else:
                kwarg_list.append({})
        return self.model(x,aux_list = kwarg_list)



####CONTROL MODEL STUFF:
    def make_positional_encoding(d_model: int, max_len: int = 5000):

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = th.sin(position * div_term)
        pe[:, 0, 1::2] = th.cos(position * div_term)
        return pe


class FF_Encoder_Layer_Component(nn.Module):
    def __init__(self, key_dim, embed_dim, hidden_layer_dim=1024):
        super().__init__()
        self.L1 = nn.Linear(embed_dim, hidden_layer_dim)
        self.act1 = nn.GLU()
        self.L2 = nn.Linear(hidden_layer_dim // 2, hidden_layer_dim)
        self.act2 = nn.GLU()
        self.KeyL = nn.Linear(hidden_layer_dim // 2, key_dim)
        self.ValueL = nn.Linear(hidden_layer_dim // 2, embed_dim)

    def forward(self, A):
        A = self.L1(A)
        A = self.act1(A)
        A = self.L2(A)
        A = self.act2(A)
        key = self.KeyL(A)
        value = self.ValueL(A)
        return value


class AttentionEncoderLayer(nn.Module):
    def __init__(self, key_dim, embed_dim, num_attention_heads):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_attention_heads, batch_first=True)
        self.key_act = PReLU(embed_dim)
        self.linear_key_out = nn.Linear(embed_dim, key_dim)
        self.value_act = PReLU(embed_dim)
        self.linear_value_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, A):
        Q = self.Wq(A)
        K = self.Wk(A)
        V = self.Wv(A)
        o = self.attention(Q, K, V)[0]
        key = self.key_act(o)
        key = self.linear_key_out(key)
        value = self.value_act(o)
        value = self.linear_value_out(value)
        return value


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(256)
        self.AttentionLayer = AttentionEncoderLayer(256, 256, 8)
        self.norm2 = nn.LayerNorm(256)

        self.FF_Layer = FF_Encoder_Layer_Component(256, 256)

    def forward(self, x, src_mask=None, is_causal=False, src_key_padding_mask=None):
        x = x + self.AttentionLayer(self.norm1(x))
        x = x + self.FF_Layer(self.norm2(x))
        return x


class TestModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(50257, 256)
        # self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,dropout=0,batch_first=True,norm_first=True),3,norm =nn.LayerNorm(256))
        self.transformer = nn.TransformerEncoder(EncoderLayer(256, 8), 3, norm=nn.LayerNorm(256))
        self.output_norm = nn.LayerNorm(256)
        self.FF1 = nn.Linear(256, 2048)
        self.Act1 = nn.GLU()
        self.FF2 = nn.Linear(1024, 2048)
        self.Act2 = nn.GLU()
        self.FF3 = nn.Linear(1024, 4)
        self.PE = nn.Parameter(make_positional_encoding(256))

    def forward(self, x):
        x = self.embed(x)
        # PE = th.permute(self.PE[:x.size(1)],[1,0,2]).repeat([x.size(0), 1, 1])
        # x += PE
        h = self.transformer(x)
        # h = x
        xdims = list(x.size())
        h = h.view(xdims[0], xdims[1], 256)
        h = th.squeeze(F.max_pool1d(th.permute(h, [0, 2, 1]), kernel_size=xdims[1], stride=1))
        h = self.output_norm(h)
        h = self.FF1(h)
        h = self.Act1(h)
        h = self.FF2(h)
        h = self.Act2(h)
        h = self.FF3(h)
        return h