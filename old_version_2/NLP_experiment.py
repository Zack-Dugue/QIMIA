import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from NLP import NLP_TextModel
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import io
import torch as th
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from transformers import PreTrainedTokenizer , AutoTokenizer,GPT2Tokenizer

import torchtext as tt
import time


def data_process(raw_text_iter):
    data = []

    for item in raw_text_iter:
        tokens = tokenizer(item)
        data.append(torch.tensor([vocab[token] for token in tokens],
                                 dtype=torch.long))
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def make_datasets(batch_size, eval_batch_size):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.get_vocab()

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)
    return train_data, val_data, test_data

def get_batch(source, i,bptt = 35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def train_for_epoch(train_data, model, optimizer, scheduler, bptt, ntokens, epoch = 0, device = None):
    criterion = nn.CrossEntropyLoss()
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source,bptt,ntokens):
    criterion = nn.CrossEntropyLoss()
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = eval_model.generate_square_subsequent_mask(bptt)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = eval_model.generate_square_subsequent_mask(data.size(0))
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)




def experiment(model, optimizer, scheduler, bptt, ntokens, batch_size, epoch = 0):
    best_val_loss = float("inf")
    epochs = 3  # The number of epochs
    train_data, val_data, test_data = make_datasets(batch_size,batch_size)
    best_model = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_for_epoch(train_data, model, optimizer, scheduler, bptt, ntokens, epoch = epoch)
        val_loss = evaluate(model, val_data,bptt,ntokens)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()
    print("training ended:")
    val_loss = evaluate(model, test_data, bptt, ntokens)
    print(f"final test loss is:{val_loss}")

LR = .01
LR_DECAY = .99
BPTT = 36
BSZ = 64

if __name__ == '__main__':
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in train_iter:
        counter.update(tokenizer(line))
    vocab = Vocab(counter)
    ntokens = 50257


    model = NLP_TextModel(256,256,ntokens,ntokens,8)
    optimizer = th.optim.Adam(model.parameters(),LR)
    scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, LR_DECAY)
    bptt = BPTT
    batch_size = BSZ


    experiment(model, optimizer, scheduler, bptt, ntokens, batch_size, epoch=0)