import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from container import QIMIA , LearnedQueryAttention
class ConvolutionBlock(nn.Module):
    def __init__(self,in_channels,out_channels, stride=1,bottle_neck = 4):
        super(ConvolutionBlock, self).__init__()
        self.block_1 = nn.Conv2d(in_channels,in_channels//bottle_neck,1, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d()
        self.act1 = nn.GELU()
        self.block_2 = nn.Conv2d(in_channels//bottle_neck,out_channels//bottle_neck,3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d()
        self.act2 = nn.GELU()
        self.block_3 = nn.Conv2d(out_channels,out_channels,1,stride=1,padding=1)

class CNNKeyProject(nn.Module):
    def __init__(self, value_dim,key_dim):
        super(CNNKeyProject, self).__init__()
        self.keyconv = nn.Conv2d(value_dim,key_dim,3,padding=1, stride=2)
        self.global_max_pool = nn.AdaptiveMaxPool2d([1,1])
        self.layer_norm = nn.LayerNorm(key_dim)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(key_dim,key_dim)

    def forward(self,x):
        x = self.keyconv(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x

def make_ConvKimia(n_blocks,channels,key_dim,n_heads):
        blocks = nn.ModuleList()
        key_proj = nn.ModuleList()
        value_proj = nn.ModuleList()
        attention = nn.ModuleList()

        for i in range(n_blocks):
            key_proj.append(CNNKeyProject(channels,key_dim))
            value_proj.append(nn.Conv1d(64,64,1))
            blocks.append(ConvolutionBlock(channels,channels))
            attention.append(LearnedQueryAttention(key_dim,n_heads))
        return QIMIA(blocks, key_proj, value_proj, CNNKeyProject(channels, key_dim), nn.Conv1d(64, 64, 1), attention)

class ConvValueOut(nn.Module):
    def __init__(self, num_blocks,num_channels):
        super(ConvValueOut, self).__init__()
        self.num_blocks = num_blocks
        self.norms = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(2,2)
        for i in range(num_blocks):
            self.norms.append(nn.BatchNorm2d(num_channels))

    def forward(self,x):
        x = self.max_pool(x)
        for i in range(self.num_blocks):
            x[:,i,:,:,:] = self.norms[i](x)




# Layerwise Kimia CNN
class CNN_L_Kimia(nn.Module):
    def __init__(self):
        super(CNN_L_Kimia, self).__init__()
        self.encoder_conv = nn.Conv2d(3,64,7,2)
        self.pool = nn.MaxPool2d(2,2)

        self.blocks1 = make_ConvKimia(6,64,32,4)
        self.blocks2 = make_ConvKimia(6,64,32,4)


