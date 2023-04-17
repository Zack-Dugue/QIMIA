import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from container import LearnedQueryAttention

class LatentIntializer(nn.Model):
    def __init__(self,latent_channels):


class UniversalLearner(nn.Module):
    def __init__(self, block,  latent_channels, num_iterations, query_proj, key_proj, value_proj,latent_initializer = None, encode=nn.Identity(),decode=nn.Identity()):
        super(UniversalLearner, self).__init__()
        self.block = block
        self.query_proj = query_proj
        self.key_proj= key_proj
        self.value_proj = value_proj

        self.attention = LearnedQueryAttention(latent_channels,8)
        self.encode = encode
        self.decode = decode
        self.block = block
        self.latent_initializer = latent_initializer
        self.T = num_iterations

    def forward(self,x):
        if self.latent_initializer is not None:

        for t in range(num_iterations):

