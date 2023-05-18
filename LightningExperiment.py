import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import math
import torchvision as tv
from torch.utils.data import DataLoader
from Vision import QIMIA_ViT
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
import os
import argparse
from pytorch_lightning.loggers import TensorBoardLogger

DATALOADER_DICT = {}
MODEL_DICT = {}

class ExperimentModule(pl.LightningModule):
    def __init__(self, model, loss_fun,optimizer):
        super.__init__()
        self.model = model
        self.loss_fun = loss_fun
        self.optimizer = optimizer
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fun(y,y_hat)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)



def experiment(path, model_name, num_nodes, num_dataloaders, batch_size, learning_rate, num_epochs, gpus, model_type,data_type):
    # accelerator = "cuda"
    accelerator = "auto"
    devices = gpus
    gpus = "auto"
    strategy = pl.strategies.DDPStrategy(static_graph=False)
    # strategy = "auto"
    # profiler = PyTorchProfiler(dirpath=path, filename='perf-logs')
    profiler = None
    logger = TensorBoardLogger(os.path.join(path, 'tb_logs'), name=model_name)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=num_epochs, strategy=strategy,
                         num_nodes=num_nodes, log_every_n_steps=50, default_root_dir=path, profiler=profiler,
                         logger=logger)
    train_loader = None
    validation_loader = None

    model = None


    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    print("Trainer system parameters:")
    print(f"\t trainer.world_size : {trainer.world_size}")
    print(f"\t trainer.num_nodes : {trainer.num_nodes}")
    print(f"\t trainer.accelerator : {trainer.accelerator}")
    print(f"\t trainer.device_ids {trainer.device_ids}")
    print(f"\t train_loader.num_workers : {train_loader.num_workers}")
    trainer.fit(ExperimentModule(model, optimizer), train_loader, validation_loader)
    print("Training run complete")
    th.save(trainer.model.state_dict(), os.path.join(path, model_name + ".pth"))
    print("Model Saved, experiment complete.")


# The Arguments of the Command Line are the following:
# Path, Model_Name, Number of Nodes, Number of Dataloaders, Batch Size, Learning Rate, Number of Epochs

if __name__ == "__main__":

    print("Running Experiment: ")
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', help='Name of experiment folder', default='test_1', type=str)
    parser.add_argument('-N', '--name', help='Name of the model', default='MODEL_1', type=str)
    parser.add_argument('-n', '--num_nodes', help='Number of nodes being run on', default=1, type=int)
    parser.add_argument('-D', '--data_type', help='Data Type', default='default', type=str)
    parser.add_argument('-M', '--model_type', help='Model Type', default='default', type=str)

    parser.add_argument('-d', '--num_dataloaders', help='Number of dataloader workers', default=1, type=int)
    parser.add_argument('-b', '--batch_size', help='Batch size', default=512, type=int)
    parser.add_argument('-l', '--learning_rate', help='Learning rate of model', default=.001, type=float)
    parser.add_argument('-e', '--num_epochs', help='Number of epochs', default=20, type=int)
    parser.add_argument('-g', '--num_gpus', help='Number of gpus per node', default=4, type=int)
    args = parser.parse_args()

    path = os.path.join(os.getcwd(), 'experiments', args.path)
    model_name = args.name
    model_type = args.model_type
    data_type = args.data_type
    num_nodes = args.num_nodes
    num_dataloaders = args.num_dataloaders
    batch_size = args.batch_size
    lr = args.learning_rate
    NumEpochs = args.num_epochs
    gpus = args.num_gpus

    print(f"Model Name: {model_name} \t num_nodes: {num_nodes} \t num_dataloaders: {num_dataloaders}"
          f"\n batch_size: {batch_size} \t learning_rate: {lr} \t num_epochs: {NumEpochs}")
    try:
        os.mkdir(path)
    except FileExistsError:
        if len(os.listdir(path)) == 0:
            pass
        else:
            pass
            # raise FileExistsError
    print(f"Experiment Info and Files stored in:{path}")
    model_name = "fred"
    experiment(path, model_name, num_nodes, num_dataloaders, batch_size, lr, NumEpochs, gpus,model_type,data_type)