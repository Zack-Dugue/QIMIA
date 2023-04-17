import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from container import QIMIA , QIMIA2, LearnedQueryAttention
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

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, stride=1, padding =1 , hidden_dim=None,dropout = 0):
        super(ConvBlock2,self).__init__()
        #This uses the reverse bottleneck formulation from convnext
        if hidden_dim == None:
            hidden_dim = in_channels*4
        self.conv1 = nn.Conv2d(in_channels, in_channels,size,stride=stride,padding=padding)
        self.act1 = nn.PReLU(in_channels)
        #idk if this batch norm is a good idea
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels,hidden_dim,1)
        # The other convolutions and non linearities are left off
        # becaues they'll be accounted for in the Value and Key projections
    def forward(self,x):
        x = th.permute(x,[0,3,1,2])
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = th.permute(x,[0,2,3,1])

        return x
def make_ConvQIMIASection(embed_dim,key_dim, num_layers,num_heads, hidden_dim = 1024):
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
        blocks.append(ConvBlock2(embed_dim,embed_dim,hidden_dim=hidden_dim))
        key_project.append(nn.Sequential(nn.PReLU(hidden_dim),nn.Linear(hidden_dim,key_dim)))
        value_project.append(nn.Sequential(nn.PReLU(hidden_dim),nn.Linear(hidden_dim,embed_dim)))

        mia = LearnedQueryAttention(key_dim,num_heads,v_dim=embed_dim, w0= True)
        memory_attention.append(mia)

    return QIMIA2(blocks,key_project,value_project,memory_attention)

class ConvDownsample(nn.Module):
    def __init__(self,num_tokens,num_channels, affine = True, PReLU = False):
        """
        Takes in an input of dim Bsz, Num Tokens, Channel width
        And returns a tensor of Bsz/4,Num tokens, Channel width
        representing a properly downsampled image
        """
        super(ConvDownsample, self).__init__()

        self.downsample = nn.MaxPool2d(2,2)
        self.num_tokens = num_tokens
        self.num_channels = num_channels
        if affine:
            self.affine = nn.Conv2d(num_tokens*num_channels*2,num_tokens*num_channels*2,1,groups=num_tokens*num_channels)
        else:
            self.affine = nn.Identity()
        if PReLU:
            self.act = nn.PReLU(num_tokens*num_channels*2)
        else:
            self.act = nn.Identity
    def forward(self,x,image_dims):
        # the way I handle the dimensionality changes here is probably inefficient
        x = x.view([-1] + [x.size(1)] + image_dims + [x.size(-1)])
        # The dimensions of x should now be:
            #bsz , num_tokens, Height, width, channels
        bsz, num_tokens, height, width, channels = x.size()
        x = th.permute(x,[0,1,4,2,3])
        x = self.downsample(x)
        x = x.view([x.size(0), x.size(2), x.size(3), x.size(1)*x.size(4)])
        x = x.expand([-1,-1,-1,x.size(3)*2])
        x = self.affine(x)
        x = self.act(x)
        x = x.view([x.size(0)*x.size(1)*x.size(2), self.num_tokens,self.num_channels*2])
        return x

class ConvGlobalAvgPool(nn.Module):
    def __init__(self):
        #performs global average pooling over the image
        # and returns it in the token form
        super(ConvGlobalAvgPool,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)


    def forward(self,x,image_dims):
        x = x.view([-1] + [x.size(1)] + [image_dims] + [x.size(-1)])
        # The dimensions of x should now be:
            #bsz , num_tokens, Height, width, channels
        x = th.permute(x[0,1,4,2,3])
        x = self.squeeze(self.pool(x))
        return x

#Specifically for image classificiation
# Based on Resnet50 Architecture.
class ConvQIMIA(nn.Module):
    def __init__(self, output_dim):
        super(ConvQIMIA, self).__init__()

        self.conv0 = nn.Conv2d(3,64,7,stride = 2,padding=3)
        self.pool = nn.MaxPool2d(2,2)
        self.norm0 = nn.BatchNorm2d(64)
        self.keyconv = nn.Conv2d(64,64,3,padding=1)
        self.valueconv = nn.Conv2d(64,64,3,padding=1)
        self.segment1 = make_ConvQIMIASection(64,64,3,4,hidden_dim=64*4)
        self.value_downsample1 = ConvDownsample(4,64)
        self.key_downsample1 = ConvDownsample(4,64)
        self.segment2 = make_ConvQIMIASection(128,128,8,8,hidden_dim=128*4)
        self.value_downsample2 = ConvDownsample(12,128)
        self.key_downsample2 = ConvDownsample(12,128)
        self.segment3 = make_ConvQIMIASection(256,256,36,16,hidden_dim=1024)
        self.value_downsample3 = ConvDownsample(48,256)
        self.key_downsample3 = ConvDownsample(48,256)
        self.segment4 = make_ConvQIMIASection(512,512,3,16,hidden_dim=1024)
        self.global_avg_pool = ConvGlobalAvgPool()
        self.output_attention = LearnedQueryAttention(512,16,v_dim = 512, w0=True)
        self.FF1 = nn.Linear(512,2048)
        self.act1 = nn.GELU(2048)
        self.FF2 = nn.Linear(2048,2048)
        self.act2 = nn.GELU(2048)
        self.FF3 = nn.Linear(2048,output_dim)

    def forward(self,x):
        x = self.conv0(x)
        x = self.pool(x)
        x = self.norm0(x)
        height = x.size(2)
        width = x.size(3)
        keys = self.keyconv(x)
        values = self.valueconv(x)
        #Ugh, dimensionality stuff is hard af.
        # current dims are
        # bz, channel, height, width
        keys = th.permute(keys, [0,2,3,1])
        keys = th.flatten(keys,0,2)
        keys = keys.view([keys.size(0),1,keys.size(1)])
        values = th.permute(values, [0,2,3,1])
        values = th.flatten(values,0,2)
        values = values.view([values.size(0),1,values.size(1)])

        keys , values = self.segment1(keys,values, [height,width, 64])
        keys = self.key_downsample1(keys, [height,width])
        values = self.values_downsample1(values,[height,width])

        keys , values = self.segment2(keys,values, [height//2,width//2, 128])
        keys = self.key_downsample2(keys, [height//2,width])
        values = self.values_downsample2(values,[height//2,width//2])

        keys , values = self.segment3(keys,values, [height//4,width//4, 256])
        keys = self.key_downsample3(keys, [height//4,width//4])
        values = self.values_downsample3(values,[height//4,width//4])

        keys , values = self.segment4(keys,values, [height//8,width//8, 512])
        keys = self.global_avg_pool(keys,[height//8,width//8])
        values = self.global_avg_pool(values,[height//8,width//8])
        o = self.output_attention(keys,values)
        o = self.FF1(o)
        o = self.act1(o)
        o = self.FF2(o)
        o = self.act2(o)
        o = self.FF3(o)
        return o



