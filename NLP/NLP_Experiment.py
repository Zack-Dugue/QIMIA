import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
import math
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from NLP_Dataloader import make_mlm_wikitext_dataloaders, make_gpt_wikitext_dataloaders, make_gpt_tinystories_dataloader
from NLP_Models import Test_Transformer, QIMIA_Transformer
from transformers import AutoTokenizer

VOCAB_SIZE = 50257


class GPT_NLP_module(pl.LightningModule):
    def __init__(self,model,loss_fun,optimizer, tokenizer):
        super().__init__()
        self.model = model
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    def training_step(self,batch,idx):
        x, y = batch
        logits = self.model(x,causal = True)
        y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits.view(-1, logits.size(-1)), y.view(-1, y.size(-1)))
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch,idx):
        x, y = batch
        logits = self.model(x, causal = True)
        y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits.view(-1, logits.size(-1)), y.view(-1, y.size(-1)))
        self.log("validation_loss", loss)

        return loss

    def test_step(self, batch,idx):
        x, y = batch
        logits = self.model(x, causal=True)
        y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits.view(-1, logits.size(-1)), y.view(-1, y.size(-1)))
        self.log("test_loss", loss)

        return loss
    def generate(self,length):
        num = self.tokenizer.encode("the")

        text = th.ones(1,length).int()*num[0]
        print(f"old text: {self.tokenizer.batch_decode(text)}")
        for i in range(1,length-1):
            guess = th.squeeze(self.model(text))
            guess = th.argmax(guess[i])
            text[0,i+1] = guess
        text = self.tokenizer.batch_decode(text)

        return text
    def on_validation_epoch_end(self):
        text = self.generate(300)
        new_text = ""
        for word in text:
            new_text = new_text + word
        print(f"\n\n generation: {new_text} \n\n")
    def configure_optimizers(self):
        return self.optimizer


class MLM_NLP_module(pl.LightningModule):
    def __init__(self,model,loss_fun,optimizer, tokenizer):
        super().__init__()
        self.model = model
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    def training_step(self,batch,idx):
        x, y, predict = batch
        logits = self.model(x,causal=False)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits[predict], y[predict])
        self.log("test_loss", loss)

        return loss


    def validation_step(self, batch,idx):
        x, y, predict = batch
        logits = self.model(x,causal=False)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits[predict], y[predict])
        self.log("validation_loss", loss)

        return loss

    def test_step(self, batch,idx):
        x, y, predict = batch
        logits = self.model(x,causal=False)
        # y = F.one_hot(y, VOCAB_SIZE).float()
        loss = self.loss_fun(logits[predict], y[predict])
        self.log("test_loss", loss)

        return loss


    def on_validation_epoch_end(self):
        # text = self.generate(300)
        # new_text = ""
        # for word in text:
        #     new_text = new_text + word
        # print(f"\n\n generation: {new_text} \n\n"
        pass
    def configure_optimizers(self):
        return self.optimizer


BSZ = 16
SEQ_LEN = 64
def GPT_experiment():
    # model = Test_Transformer(8,VOCAB_SIZE,128,8)
    model = QIMIA_Transformer(256,256,VOCAB_SIZE,VOCAB_SIZE,8)
    optimizer = th.optim.Adam(model.parameters(),.01)
    module = GPT_NLP_module(model, nn.CrossEntropyLoss(),optimizer,    tokenizer = AutoTokenizer.from_pretrained('gpt2'))
    generation = module.generate(30)
    print(f"text generation {generation}" )
    my_trainer = pl.Trainer()
    train_dataloader, validation_dataloader , test_dataloader = make_gpt_wikitext_dataloaders(BSZ,SEQ_LEN)
    my_trainer.fit(module,train_dataloader,validation_dataloader)


def MLM_experiment():
    model = Test_Transformer(8,VOCAB_SIZE+1,128,8)
    # model = QIMIA_Transformer(256,256,VOCAB_SIZE+1,VOCAB_SIZE,8)
    optimizer = th.optim.Adam(model.parameters(),.01)
    module = MLM_NLP_module(model, nn.CrossEntropyLoss(),optimizer,    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased'))
    # generation = module.generate(30)
    # print(f"text generation {generation}" )
    my_trainer = pl.Trainer()
    train_dataloader, validation_dataloader , test_dataloader = make_mlm_wikitext_dataloaders(BSZ,SEQ_LEN)
    my_trainer.fit(module,train_dataloader,validation_dataloader)



if __name__ == '__main__':
    # MLM_experiment()
    GPT_experiment()