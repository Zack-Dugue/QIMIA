import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from Implementation import BaseBlock, InitializerBlock, OutputBlock , TransformBlock, QIMIA_Sequential
from transformers import AutoTokenizer, PreTrainedTokenizer
import math



# The lack of a channels last memory format here is pretty brutal. Requires me to alter the memory like
# every single forward pass of every block.
# This is because inherently, we're cramming the dimensions from the image itself into the batch dimension.
# Normally this is efficient and smart, but this makes it so that, in order to get a contiguous view
# the "channel dim" must ALWAYS BE THE LAST DIMENSION. To try and view or reshape or get any parts of
# dim1 (which is the crammed batch dimension) to come after the channel dim
# IE of the from (crammed_batch_dim, channel_dim, other_dim), is non-contiguous in this implementation.
# and swapping back results in something non contiguous as well (and thus convolutions won't work).
# This results in a CONSIDERABLE efficient hit, because we have to essentially remake the memory (by calling .contiguous) before
# and after every CONV block. And we can't just switch to Channel first QIMIA, because that wouldn't work
# with the attentino implementation in Pytorch.
# So we need to find a Channel Last implementation of convolutions for pytorch,
# which apparently do exist .
# https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
class ConvBlock2(BaseBlock):
    def __init__(self,key_dim,embed_dim, size=3, stride=1, padding =1 , hidden_dim=None,dropout = 0):
        super(ConvBlock2,self).__init__(key_dim,embed_dim)
        #This uses the reverse bottleneck formulation from convnext
        if hidden_dim == None:
            hidden_dim = embed_dim*4
        self.conv1 = nn.Conv2d(embed_dim, embed_dim,size,stride=stride,padding=padding)
        self.act1 = nn.PReLU(embed_dim)
        #idk if this batch norm is a good idea
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.conv2 = nn.Conv2d(embed_dim,hidden_dim,1)
        self.act_key = nn.PReLU(hidden_dim)
        self.act_value = nn.PReLU(hidden_dim)
        self.conv_key = nn.Conv2d(hidden_dim,key_dim)
        self.conv_value = nn.Conv2d(hidden_dim,embed_dim)
        # The other convolutions and non linearities are left off
        # becaues they'll be accounted for in the Value and Key projections
    def forward(self,x, image_dims=None):
        if image_dims == None:
            raise ValueError("image_dims must be given for convolutional block")
        x.view([-1] + image_dims + x.size(1))
        x = th.permute([0,3,1,2]).contiguous()
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        key = self.act_key(x)
        key = self.conv_key(x)
        value = self.act_key(x)
        value = self.conv_value(x)

        key = th.permute(key,[0,2,3,1])
        value = th.permute(value,[0,2,3,1])

        return key,value

class ConvInitializer(InitializerBlock):
    def __init__(self,key_dim, embed_dim, kernel_size=7,stride=2,padding=3,base_dim = None,pool=True):
        super().__init__()
        if base_dim == None:
            base_dim = embed_dim
        self.conv0 = nn.Conv2d(3,base_dim,kernel_size,stride=stride,padding=padding)
        if pool:
            self.max_pool = nn.MaxPool2d()
        else:
            self.max_pool = nn.Identity()
        self.bn0 = nn.BatchNorm2d(base_dim)
        self.conv1 = nn.Conv2d(base_dim,base_dim,3,padding=1)
        self.key_act = nn.PReLU(base_dim)
        self.key_conv = nn.Conv2d(base_dim,key_dim,padding=1)
        self.value_act = nn.PReLU(base_dim)
        self.value_conv = nn.Conv2d(base_dim,embed_dim,padding=1)

    def forward(self,x):
        x = self.conv0(x)
        x = self.max_pool(x)
        x = self.bn0(x)
        x = self.conv1(x)
        key = self.key_act(x)
        key = self.key_conv(key)
        key = th.permute(key,[0,2,3,1])
        value = self.value_act(x)
        value = self.value_conv(value)
        value = th.permute(value,[0,2,3,1])

        return key, value

class DownsampleBlock(TransformBlock):
    def __init__(self,key_dim_0,embed_dim_0, key_dim_f,embed_dim_f,num_tokens, pool_type='max'):
        super().__init__()
        # for now this assumes the f dims are a multiple of the initial dims.
        self.pool_type = pool_type
        self.value_norm = nn.BatchNorm2d(embed_dim_0)
        self.value_act = nn.PReLU(embed_dim_0)
        self.value_affine = nn.Conv2d(embed_dim_0*num_tokens, embed_dim_f*num_tokens,1,groups=embed_dim_f)

        self.key_norm = nn.BatchNorm2d(key_dim_0)
        self.key_act = nn.PReLU(key_dim_0)
        self.key_affine = nn.Conv2d(key_dim_0 * num_tokens, key_dim_f * num_tokens, 1, groups=key_dim_f)

    def block(self,keys,values,image_dims=None):
        if image_dims == None:
            raise ValueError("Must give kwarg for image_dims in DownsampleBlock")
        old_key_bsz = keys.size(0)
        old_val_bsz = values.size(0)
        old_key_token_dim = keys.size(1)
        old_val_token_dim = values.size(1)
        keys = th.flatten(keys,1,2)
        keys = keys.view([-1] + image_dims + [keys.view(1)])
        keys = th.permute(keys, [0,3,1,2])
        values = th.flatten(values,1,2)
        values = values.view([-1] + image_dims + [values.view(1)])
        values = th.permute(values, [0,3,1,2])
        if self.pool_type == "average":
            keys = F.avg_pool2d(keys,image_dims)
            values = F.avg_pool2d(values,image_dims)
        elif self.pool_type == "max":
            MAX = F.max_pool2d(keys, image_dims)
            values = F.max_pool2d(values, image_dims)
        keys = self.key_norm(keys)
        keys = self.key_act(keys)
        keys = self.key_affine(keys)
        values = self.value_norm(values)
        values = self.value_act(values)
        values = self.value_affine(values)
        keys = th.permute(keys,[0,2,3,1])
        # keys = keys.view([old_key_bsz] + old_to)
        return keys , values
