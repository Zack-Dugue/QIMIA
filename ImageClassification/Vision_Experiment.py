import torch as th
import torch.nn as nn
import torch.nn.functional  as F

from Vision_Models import QIMIA_ViT
import argparse

from Vision_Dataloders import get_CIFAR10_dataloader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os

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
        print(f"\rtrain_loss : {loss}", end="")
        return loss

    def validation_step(self, batch,idx):
        x, y = batch
        logits = self.model(x)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits, y)
        print(f"\rvalidation_loss : {loss}", end="")

        return loss

    def test_step(self, batch,idx):
        x, y = batch
        logits = self.model(x)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits, y)
        print(f"\rtest_loss : {loss}", end="")

        return loss

    def configure_optimizers(self):
        return self.optimizer

LR = .1
def experiment(path, model_name, num_nodes, num_dataloader_workers, batch_size, learning_rate, num_epochs, gpus):
    if gpus == 0:
        accelerator = "cpu"
        devices = "auto"
    else:
        accelerator = "cuda"
        devices = gpus
    if gpus > 1:
        strategy = pl.strategies.DDPStrategy(static_graph=True)
    else:
        strategy = "auto"
    # strategy = "auto"
    # profiler = PyTorchProfiler(dirpath=path, filename='perf-logs')
    profiler = None
    logger = TensorBoardLogger(os.path.join(path, 'tb_logs'), name=model_name)
    train_loader, val_loader, test_loader = get_CIFAR10_dataloader(64,batch_size,num_dataloaders=num_dataloaders)
    model = QIMIA_ViT(768,768,224,16,1000,12,input_attention_heads =8 , FF_hidden_dim = 3072, output_hidden_dim=3072)
    optimizer = th.optim.Adam(model.parameters(), learning_rate)
    module = Image_Classification_Module(model,nn.CrossEntropyLoss(),optimizer=optimizer)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=num_epochs, strategy=strategy,
                         num_nodes=num_nodes, log_every_n_steps=50, default_root_dir=path, profiler=profiler,
                         logger=logger)
    print(f"Number of model params: {num_params}")
    print("Trainer system parameters:")
    print(f"\t trainer.world_size : {trainer.world_size}")
    print(f"\t trainer.num_nodes : {trainer.num_nodes}")
    print(f"\t trainer.accelerator : {trainer.accelerator}")
    print(f"\t trainer.device_ids {trainer.device_ids}")
    print(f"\t train_loader.num_workers : {train_loader.num_workers}")
    # model = tv.models.VisionTransformer(64,16,12,12,768,3072)
    # num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of default VIT params : {num_params}")
    trainer.fit(module,train_dataloaders=train_loader,val_dataloaders=val_loader)


#taken from my CS156b code. PRimarily written by (guy who's name I forgot unfortunately but who was very cool,
# and a great programmer).
if __name__ == "__main__":

    print("Running Experiment: ")
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', help='Name of experiment folder', default='test_1', type=str)
    parser.add_argument('-N', '--name', help='Name of the model', default='MODEL_1', type=str)
    parser.add_argument('-n', '--num_nodes', help='Number of nodes being run on', default=1, type=int)
    parser.add_argument('-d', '--num_dataloaders', help='Number of dataloader workers', default=1, type=int)
    parser.add_argument('-b', '--batch_size', help='Batch size', default=36, type=int)
    parser.add_argument('-l', '--learning_rate', help='Learning rate of model', default=.001, type=float)
    parser.add_argument('-e', '--num_epochs', help='Number of epochs', default=20, type=int)
    parser.add_argument('-g', '--num_gpus', help='Number of gpus per node', default=0, type=int)
    args = parser.parse_args()

    path = os.path.join(os.getcwd(), 'experiments', args.path)
    model_name = args.name
    num_nodes = args.num_nodes
    num_dataloaders = args.num_dataloaders
    batch_size = args.batch_size
    lr = args.learning_rate
    NumEpochs = args.num_epochs
    gpus = args.num_gpus

    print(f"Model Name: {model_name} \t num_nodes: {num_nodes} \t num_dataloaders: {num_dataloaders}"
          f"\n batch_size: {batch_size} \t learning_rate: {lr} \t num_epochs: {NumEpochs}")

    experiment(path, model_name, num_nodes, num_dataloaders, batch_size, lr, NumEpochs, gpus)
