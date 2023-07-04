import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import math
import torchvision as tv
from torch.utils.data import DataLoader
from Vision_Models import QIMIA_ViT
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from Vision_Dataloders import get_CIFAR10_dataloader
import pytorch_lightning as pl

class Image_Classification_Module(pl.LightningModule):
    def __init__(self,model,loss_fun,optimizer):
        super().__init__()
        self.model = model
        self.loss_fun = loss_fun
        self.optimizer = optimizer

    def training_step(self,batch,idx):
        x, y = batch
        logits = self.model(x)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits, y)
        return loss

    def validation_step(self, batch,idx):
        x, y = batch
        logits = self.model(x)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits, y)
        return loss

    def test_step(self, batch,idx):
        x, y = batch
        logits = self.model(x)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits, y)
        return loss

    def configure_optimizers(self):
        return self.optimizer

LR = .1
def experiment():
    train_loader, val_loader, test_loader = get_CIFAR10_dataloader(64,64)
    model = QIMIA_ViT(512,256,64,16,10,12,input_attention_heads =8 , FF_hidden_dim = 1024, output_hidden_dim=1024)
    optimizer = th.optim.Adam(model.parameters(), LR)
    module = Image_Classification_Module(model,nn.CrossEntropyLoss(),optimizer=optimizer)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of  params: {num_params}")
    # model = tv.models.VisionTransformer(64,16,12,12,768,3072)
    # num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of default VIT params : {num_params}")
    trainer = pl.Trainer()
    trainer.fit(module,train_dataloaders=train_loader,val_dataloaders=val_loader)

if __name__ == "__main__":
    experiment()