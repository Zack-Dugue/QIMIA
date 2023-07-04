import torch as th
import torch.nn as nn
import torch.nn.functional as F
from Implementation import *
import matplotlib.pyplot as plt
import numpy as np

class MLPBlock(BaseBlock):
    def __init__(self, key_dims, value_dims, hidden_dim):
        super(MLPBlock, self).__init__(key_dims, value_dims)
        self.fc1 = nn.Linear(value_dims, hidden_dim)
        self.act1 = nn.GELU()
        self.fcValue = nn.Linear(hidden_dim, value_dims)
        self.fcKey = nn.Linear(hidden_dim, key_dims)
    def block(self,A):
        A = self.fc1(A)
        A = self.act1(A)
        value = self.fcValue(A)
        key = self.fcKey(A)
        return key,value

class SabotageBlock(BaseBlock):
    def __init__(self, key_dims, value_dims, hidden_dim):
        super(SabotageBlock, self).__init__(key_dims, value_dims)
        self.key_dims = key_dims
        self.value_dims = value_dims
    def block(self,A):
        bsz = A.size(0)
        return th.randn(bsz, self.key_dims), th.randn(bsz, self.value_dims)

class InvisibleBlock(BaseBlock):
    def __init__(self, key_dims, value_dims, hidden_dim):
        super(InvisibleBlock, self).__init__(key_dims, value_dims)
        self.key_dims = key_dims
        self.value_dims = value_dims
    def block(self,A):
        bsz = A.size(0)
        return -1000000*th.ones(bsz, self.key_dims), -1000000*th.ones(bsz, self.value_dims)


class Initializer(InitializerBlock):
    def __init__(self, key_dims, value_dims, input_dim):
        super(InitializerBlock, self).__init__()
        self.fcValue = nn.Linear(input_dim,value_dims)
        self.fcKey = nn.Linear(input_dim,key_dims)

    def block(self,A):
        key = self.fcKey(A)
        value = self.fcValue(A)
        return key,value

class Output(OutputBlock):
    def __init__(self, key_dims, value_dims):
        super(Output, self).__init__(key_dims, value_dims)
        self.fcOut = nn.Linear(value_dims, 1)
        self.outact = nn.Sigmoid()
    def block(self,A):
        A = self.fcOut(A)
        A = self.outact(A)
        return th.squeeze(A)

class BasicMLP(nn.Module):
    def __init__(self,num_blocks,input_dim,value_dim,hidden_dim = 20,skip_connection=False):
        super().__init__()
        block_list = []

        for block in range(num_blocks-2):
            block_list.append(nn.Sequential(nn.LayerNorm(value_dim), nn.Linear(value_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,value_dim)))
        self.init_block = nn.Linear(input_dim,value_dim)
        self.output_block = nn.Sequential(nn.LayerNorm(value_dim), nn.Linear(value_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,1), nn.Sigmoid())
        self.blocks = block_list
        self.skip_connection = skip_connection

    def forward(self,x):
        x = self.init_block(x)
        for block in self.blocks:
            if self.skip_connection:
                x = x + block(x)
            else:
                x = block(x)
        return self.output_block(x)




def experiment(iterations, key_dims, value_dims, input_dim=10, hidden_dim=20):
    model = QIMIA_Sequential([Initializer(key_dims, value_dims, input_dim),MLPBlock(key_dims,value_dims,hidden_dim),
                      MLPBlock(key_dims,value_dims,hidden_dim), MLPBlock(key_dims,value_dims,hidden_dim),SabotageBlock(key_dims,value_dims,hidden_dim),Output(key_dims,value_dims)])
    model = BasicMLP(6,10,10,20)
    bsz = 16
    loss_fun = nn.BCELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.000005)
    losses = []
    for i in range(iterations):
        y = th.randint(2,[bsz,1]).to(th.float)
        y_rep = y.repeat([1,input_dim])
        x = y_rep + th.randn_like(y_rep)
        y_hat = model(x)
        loss = loss_fun(th.squeeze(y_hat),th.squeeze(y))
        losses.append(float(loss))
        print(f"iteration: {i} loss: {loss}")
        loss.backward()
        optimizer.step()
    plt.plot(np.array(losses))
    plt.show()
    model.get_logs(path ="../old_version/logs")

if __name__ == "__main__":
    experiment(100000, 16, 16, 10, 10)
