import torch as th
import torch.nn as nn
import torch.nn.functional  as F

from Vision_Models import QIMIA_ViT , TinyModel
import argparse
import datasets
from Vision_Dataloders import get_CIFAR10_dataloader , get_imageNet_dataloader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os

class Image_Classification_Module(pl.LightningModule):
    def __init__(self,model,loss_fun,optimizer, QIMIA_log = True):
        super().__init__()
        self.model = model
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        # self.scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, .9)

        self.QIMIA_log = QIMIA_log
        self.validation_loss = []
        self.validation_accuracy = []


    def training_step(self,batch,idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fun(logits, y)
        _, predicted = th.max(logits.data, 1)
        accuracy = (predicted == y).sum() / x.size(0)
        if self.QIMIA_log:
            logs = self.model.get_logs()
            self.log('H' , logs['H'],sync_dist=True)
            self.log('R' , logs['R'],sync_dist=True)
            self.log('S' , logs['S'],sync_dist=True)
            self.log('q_norm', logs['q_norm'],sync_dist=True)
            print(f"\r training: idx = {idx} =  loss = {loss}, accuracy = {accuracy} 'H' = {logs['H']}, 'R' = {logs['R']} , 'S' = {logs['S']}, 'q_norm' = {logs['q_norm']}" , end="")
        else:
            print(f"\r training: idx = {idx} =  loss = {loss}, accuracy = {accuracy}" , end="")

        self.log("train_loss", loss,sync_dist=True)
        self.log("train_accuracy", accuracy,sync_dist=True)
        return loss

    def validation_step(self, batch,idx):
        x, y = batch
        logits = self.model(x)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits, y)
        print(f"\r Validation idx = {idx}, loss : {loss}", end="")
        _, predicted = th.max(logits.data, 1)
        accuracy = (predicted == y).sum() / x.size(0)
        self.validation_loss.append(loss)
        self.validation_accuracy.append(accuracy)
        return loss

    def test_step(self, batch,idx):
        x, y = batch
        logits = self.model(x)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits, y)
        print(f"\r test_loss : {loss}", end="")
        self.log("test_loss", loss)

        return loss
    def on_validation_epoch_end(self):
        val_loss = sum(self.validation_loss) / len(self.validation_loss)
        val_accuracy = sum(self.validation_accuracy) / len(self.validation_accuracy)

        self.log("val_loss", val_loss,sync_dist=True)
        self.log("val_accuracy", val_accuracy,sync_dist=True)
        print(f"val_loss = {val_loss} , val_accuracy = {val_accuracy} \n")
        self.validation_loss = []
        self.validation_accuracy = []
        # self.scheduler.step()

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
    print("Initializing Logger")
    logger = TensorBoardLogger('tb_logs', name=model_name)
    print("Initializing Dataloaders")
    train_loader, val_loader, test_loader = get_imageNet_dataloader(224,batch_size,num_dataloader_workers=num_dataloader_workers)
    print(f"length of dataloader: {len(train_loader)}")
    print("Initializing Model")
    # model = QIMIA_ViT(768,768,224,16,1000,12,input_attention_heads =8 , FF_hidden_dim = 3072, output_hidden_dim=3072)
    # model = QIMIA_ViT(256,256,64,16,1000,12,input_attention_heads =8 , FF_hidden_dim = 3072, output_hidden_dim=
    model = QIMIA_ViT(512, 512, 224, 16, 1000, 12, input_attention_heads=8, FF_hidden_dim=2048, output_hidden_dim=2048)
    print(f"Model Num Parameters: {model.parameters()}")
    print(f"Memory left after model initialization : {th.cuda.max_memory_allocated() }")
    print("initializing optimizer")
    optimizer = th.optim.Adam(model.parameters(), learning_rate)
    print("initializng image classification module")
    module = Image_Classification_Module(model,nn.CrossEntropyLoss(),optimizer=optimizer,QIMIA_log=True)
    print("initializing trainer")
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, max_epochs=num_epochs, strategy=strategy,
                         num_nodes=num_nodes, default_root_dir=path, profiler=profiler,
                         logger=logger)

    print("Trainer system parameters:")
    print(f"\t trainer.world_size : {trainer.world_size}")
    print(f"\t trainer.num_nodes : {trainer.num_nodes}")
    print(f"\t trainer.accelerator : {trainer.accelerator}")
    print(f"\t trainer.device_ids {trainer.device_ids}")
    print(f"\t trainer.strategy : {trainer.strategy}")
    print(f"\t train_loader.num_workers : {train_loader.num_workers}")

    # model = tv.models.VisionTransformer(64,16,12,12,768,3072)
    # num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of default VIT params : {num_params}")
    print("begin training")
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
