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

class PSeLU(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.a = nn.Parameter(th.ones(embed_dim,requires_grad=True))
        self.b = nn.Parameter(th.zeros(embed_dim,requires_grad=True))
        self.sigm = nn.Sigmoid()
    def forward(self,x):

        return self.sigm(x*self.a+self.b)*x

class VisionInitializer(InitializerBlock):
    def __init__(self, image_size, patch_size, embedding_dim, key_dim):
        super(VisionInitializer, self).__init__()

        # If you write a flexible method to chop into patches, or write the EC conv method and pass
        #  the correct optional arument to Conv2d / Conv1d, you don't need to check this:
        assert image_size % patch_size == 0, "image size must be divisible by patch size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = embedding_dim


        # Patches
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2  # RGB has 3 channels
        # Layers
        self.patch_embedding_values = nn.Linear(self.patch_dim, embedding_dim)
        self.patch_embedding_keys = nn.Linear(self.patch_dim, key_dim)

        self.position_embedding_values = nn.Parameter(th.randn(1, self.num_patches + 1, embedding_dim))
        self.position_embedding_keys = nn.Parameter(th.randn(1, self.num_patches + 1, key_dim))
        self.cls_token_value = nn.Parameter(th.randn(1, 1, embedding_dim))
        self.cls_token_key = nn.Parameter(th.randn(1, 1, key_dim))
        self.flatten_to_token = nn.Flatten(0,1)
    def chop_to_patches(self, x, b, c):
        ''' Chops each image in batch x into patches of dimensions patch_size**2.
            This method uses tensor reshaping.
            input:
              x: batch of images, with shape (batch_size, num_channels, height, width)
              b: batch_size
              c: num_channels
            output:
              chopped_x: chop-chop! shape (batch_size, num_patch_per_row, num_patch_per_col,
                                           num_channels, patch_size, patch_size)
        '''
        # - shape is now (batch_size, num_channels, num_patch_per_row, patch_size, num_patch_per_col, patch_size)
        chopped_x = x.reshape(b, c, self.image_size // self.patch_size, self.patch_size,
                      self.image_size // self.patch_size, self.patch_size)


        return chopped_x
    def block(self, x):
        # shape is (batch_size, num_channels, height, width)
        b, c, w, h = x.shape
        assert h == self.image_size and w == self.image_size, \
            f"Input image size ({h}*{w}) doesn't match model expected size ({self.image_size}*{self.image_size})"

        # Chop into patches
        x = self.chop_to_patches(x, b, c)

        # Flatten patches - recall patch_dim=(num_channels*patch_size**2)
        x = x.reshape(b, self.num_patches, self.patch_dim)

        # Linear embedding
        # TODO: Call the appropriate method to perform linear embedding
        values = self.patch_embedding_values(x)
        keys = self.patch_embedding_keys(x)

        # Add class token
        cls_token_value = self.cls_token_value.expand(x.shape[0], -1, -1)
        cls_token_key =  self.cls_token_key.expand(x.shape[0], -1, -1)
        values = th.cat((cls_token_value, values), dim=1)
        keys = th.cat((cls_token_key, keys), dim=1)
        # Add position tokens
        PE_key = self.flatten_to_token(self.position_embedding_keys[:, :(x.shape[1]+1)]).repeat([x.size(0),1])
        PE_value = self.flatten_to_token(self.position_embedding_values[:, :(x.shape[1]+1)]).repeat([x.size(0),1])
        return [self.flatten_to_token(keys), PE_key], [self.flatten_to_token(values), PE_value]

class FF_Block(BaseBlock):
    def __init__(self,key_dim,embed_dim, hidden_layer_dim = 1024):
        super().__init__(key_dim,embed_dim)
        self.L0 = nn.Linear(embed_dim, hidden_layer_dim)
        self.value_act = PSeLU(hidden_layer_dim)
        self.key_act = PSeLU(hidden_layer_dim)
        self.L_values = nn.Linear(hidden_layer_dim,embed_dim)
        self.L_keys = nn.Linear(hidden_layer_dim,key_dim)

    def block(self, A):
        A = self.L0(A)
        value = self.value_act(A)
        value = self.L_values(value)
        key = self.key_act(A)
        key = self.L_keys(key)
        return key, value

class AttentionBlock(BaseBlock):
    def __init__(self,key_dim,embed_dim, num_attention_heads):
        super().__init__(key_dim,embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_attention_heads,batch_first=True)
        self.key_act = PReLU(embed_dim)
        self.linear_key_out = nn.Linear(embed_dim,key_dim)
        self.value_act = PReLU(embed_dim)
        self.linear_value_out = nn.Linear(embed_dim,embed_dim)
    def block(self,A,num_input_tokens= None):
        if num_input_tokens == None:
            raise ValueError("Need to specify the numebr of tokens in the input to" \
                             "an attention layer for BERT_STYLE_NLP QIMIA")
        A = A.view(A.size(0)//num_input_tokens, num_input_tokens, A.size(1))

        o = self.attention(A,A,A)[0]
        key = self.key_act(o)
        key = self.linear_key_out(key)
        value = self.value_act(o)
        value = self.linear_value_out(value)
        return key , value

class VisionOutput(OutputBlock):
    def __init__(self,key_dim, embed_dim, output_classes, hidden_dim = 1024):
        super().__init__(key_dim,embed_dim)
        self.Linear_0 = nn.Linear(embed_dim,hidden_dim)
        self.act = nn.GELU()
        self.Linear_1 = nn.Linear(hidden_dim,output_classes)
    def block(self,A,num_input_tokens= None):
        A = A.view(A.size(0)//num_input_tokens, num_input_tokens, A.size(1))
        A = A[:,-1,:]
        A = self.Linear_0(A)
        A = self.act(A)
        out = self.Linear_1(A)
        return out

class QIMIA_ViT(nn.Module):
    def __init__(self,key_dim,embed_dim, image_size, patch_size, num_output_classes, num_layers, input_attention_heads = 8, FF_hidden_dim=1024,output_hidden_dim = 2048):
        super().__init__()
        blocks = [VisionInitializer(image_size, patch_size, embed_dim, key_dim)]
        for i in range(num_layers):
            blocks.append(AttentionBlock(key_dim, embed_dim, input_attention_heads))
            blocks.append(FF_Block(key_dim,embed_dim,hidden_layer_dim=FF_hidden_dim))
        blocks.append(VisionOutput(key_dim,embed_dim,num_output_classes))
        # blocks[1].LQA.diagnostic = True
        self.model = QIMIA_Sequential(blocks)
        self.num_patches = blocks[0].num_patches + 1

    def parameters(self):
        params = self.model.parameters()
        return self.model.parameters()
    def forward(self,x):
        kwarg_list = []
        for i in range(len(self.model)):
            if i % 2 == 1:
                kwarg_list.append({"num_input_tokens" : self.num_patches})
            else:
                kwarg_list.append({})
        return self.model(x,aux_list = kwarg_list)