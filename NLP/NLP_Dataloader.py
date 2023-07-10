import torch.utils.data
import torchtext as tt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Subset,DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, DataCollatorForLanguageModeling
import time
import pickle
import os
import csv
def load_Wikitext(data_root, seq_length = 10):
    trainset = tt.datasets.WikiText2(split='train')
    validation_set = tt.datasets.WikiText2(split='validation')
    test_set = tt.datasets.WikiText2(split='test')



class GPT_MyWikiText(Dataset):
    def __init__(self,split,seq_length,tokenizer):
        self.data = []
        self.seq_length = seq_length
        super(GPT_MyWikiText, self).__init__()
        proto_dataset = tt.datasets.WikiText2(split=split)

        for i, item in enumerate(proto_dataset):
            if len(item) <= seq_length:
                continue
            text = tokenizer(item)['input_ids']
            num_elements = len(text)//(seq_length+1)
            for element in range(num_elements):
                self.data.append(text[element*seq_length: element*seq_length+seq_length+1])

    def gpt_it(self,data : th.Tensor):
        data = data.repeat([self.seq_length , 1, 1])
        return data


    def __getitem__(self,idx):
        x = th.Tensor(self.data[idx][0:self.seq_length]).to(th.int64)
        y = th.Tensor((self.data[idx])[1:self.seq_length+1]).to(th.int64)
        return (x, y)
    def __len__(self):
        return len(self.data)




# Use built in hugging face data collator at somepoint
class MLM_MyWikiText(Dataset):
    def __init__(self,split,seq_length,tokenizer : PreTrainedTokenizer, alter_prob = .15, mask_prob = .8, randomize_prob = .1, remain_prob = .1):
        super(MLM_MyWikiText, self).__init__()
        self.data = []
        self.seq_length = seq_length
        assert(mask_prob + randomize_prob + remain_prob == 1)
        self.tokenizer = tokenizer
        proto_dataset = tt.datasets.WikiText2(split=split)

        for i, item in enumerate(proto_dataset):
            if len(item) <= seq_length:
                continue
            text = tokenizer(item)['input_ids']
            for j in range(len(text)-seq_length):
                assert(len(text[j:j+seq_length]) != 0)
                self.data.append(text[j:j+seq_length])
                print(f"\r i : {i} , j = {j}",end='')
        self.alter_prob = alter_prob
        self.mask_prob = mask_prob
        self.randomize_prob = randomize_prob
        self.remain_prob = remain_prob

    def mask_data(self, data):
        rand1 = th.rand_like(data)
        rand2 = th.rand_like(data)
        rand3 = th.randint_like(data,0,self.tokenizer.vocab_size)
        predict = th.where(rand1 < self.alter_prob, True, False)
        data = th.where(th.logical_and(rand2 < self.mask_prob , predict), self.tokenizer.mask_token_id, data)
        data = th.where(th.logical_and( th.logical_and((self.mask_prob + self.randomize_prob) > rand2, rand2 > self.mask_prob) , predict), rand3, data)
        return data , predict.bool()

    def __getitem__(self,idx):
        data = th.Tensor(self.data[idx])
        x,  predict = self.mask_data(data)
        y = data
        return (x.long(), y.long(), predict)
    def __len__(self):
        return len(self.data)

def make_gpt_wikitext_dataloaders(bsz, seq_len):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    train_set = GPT_MyWikiText('train',seq_len,tokenizer)
    validation_set = GPT_MyWikiText('valid',seq_len,tokenizer)
    test_set = GPT_MyWikiText('test',seq_len,tokenizer)
    train_dataloader = DataLoader(train_set,bsz,shuffle= True)
    validation_dataloader = DataLoader(validation_set,bsz,shuffle=True)
    test_dataloader = DataLoader(test_set,bsz,shuffle= True)
    return train_dataloader, validation_dataloader, test_dataloader

def make_mlm_wikitext_dataloaders(bsz, seq_len):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'mask_token': '<MASK>'})
    train_set = MLM_MyWikiText('train',seq_len,tokenizer)
    validation_set = MLM_MyWikiText('valid',seq_len,tokenizer)
    test_set = MLM_MyWikiText('test',seq_len,tokenizer)
    train_dataloader = DataLoader(train_set,bsz,shuffle= True)
    validation_dataloader = DataLoader(validation_set,bsz,shuffle=True)
    test_dataloader = DataLoader(test_set,bsz,shuffle= True)
    return train_dataloader, validation_dataloader, test_dataloader


## TODO: ): get this working. Currently the issue is that the dataset is to large and I don't have enough ram
##  So my stuff crashes. The solution could be to use datasets from https://huggingface.co/learn/nlp-course/chapter5/4?fw=pt
##  but I don't know how to actually index that or anything.


class GPT_TinyStories(Dataset):
    def __init__(self, path:str,split:str, seq_len:int, tokenizer : PreTrainedTokenizer, chache_this = True):
        """
        The Tiny Stories dataset
        :param path:
        :param split:
        :param seq_len:
        :param tokenizer:
        """
        #We need to open the data

        print("loading has begun")
        dataset = load_dataset("data/TinyStories")
        dataset = dataset.data[split][0]
        print(dataset)
        self.tokenizer = tokenizer


    def tokenize_map(self,element):
        outputs = self.tokenizer(
            element["text"],
            truncation=True,
            max_length=self.seq_len,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == self.seq_len:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}



def make_gpt_tinystories_dataloader(bsz,seq_len,tokenizer:PreTrainedTokenizer, val_split = .1):
    tokenizer.add_special_tokens({'eos_token':"<|endoftext|>"})

    test_data = GPT_TinyStories('data', 'valid',seq_len,tokenizer)
    train_dataset = GPT_TinyStories('data', 'train',seq_len,tokenizer)
    val_len = train_dataset*val_split//1
    train_data,val_data = torch.utils.data.random_split([len(train_dataset) - val_len, val_len])
    train_dataloader = DataLoader(train_data,bsz,shuffle=True)
    val_dataloader = DataLoader(val_data,bsz,shuffle=False)
    test_dataloader = DataLoader(test_data,bsz,shuffle=False)

    return train_dataloader,val_dataloader,test_dataloader



from datasets import load_dataset
import datasets
#Stolen from Huggingface Tutorial website:


if __name__ == '__main__':
    pass