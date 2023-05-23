import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
import math
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer , AutoTokenizer,GPT2Tokenizer
from NLP import NLPClassifier , NLPClassifier2, TestModel2
import matplotlib.pyplot as plt



def experiment(epochs,batch_size,lr):
    # th.cuda.empty_cache()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"current device is {device}")
    auto_tokenize = GPT2Tokenizer.from_pretrained("gpt2")
    auto_tokenize.pad_token = auto_tokenize.eos_token
    dataset = tt.datasets.AG_NEWS()[0]
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    test_model = NLPClassifier2(256,128,50257,4,4).to(device)
    # test_model = TestModel2(0).to(device)
    num_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(f"number of  params: {num_params}")
    loss_fun = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(test_model.parameters(),lr)
    th.nn.utils.clip_grad_norm_(test_model.parameters(), 2, norm_type=2.0, error_if_nonfinite=False, foreach=None)
    for epoch in range(epochs):
        i = 0
        losses = []
        for y,x in dataloader:
            i += 1
            optimizer.zero_grad()
            x = auto_tokenize(x,padding=True)['input_ids']
            x = th.LongTensor(x).to(device)
            # x = F.one_hot(x,num_classes = 50257)
            y = F.one_hot(y - th.ones_like(y),num_classes =4).float().to(device)
            y_hat = test_model(x)
            loss = loss_fun(y_hat,y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if i == 1000:
                # test_model.model.get_logs('new_logs')
                plt.plot(losses)
                plt.show()
                return
            print(f"iteration - {i}\tepoch - {epoch}\tloss - {loss}\r")
# def viz_experiment():
#     epochs, batch_size, lr = (1,7,.1)
#     auto_tokenize = GPT2Tokenizer.from_pretrained("gpt2")
#     auto_tokenize.pad_token = auto_tokenize.eos_token
#     dataset = tt.datasets.AG_NEWS()[0]
#     dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
#     test_model = NLPClassifier(256,256,50257,4,2)
#     loss_fun = nn.CrossEntropyLoss()
#     params = test_model.parameters()
#     optimizer = th.optim.SGD(params,lr)
#     for epoch in range(epochs):
#         i = 0
#         for y,x in dataloader:
#             i += 1
#             optimizer.zero_grad()
#             x = auto_tokenize(x,padding=True)['input_ids']
#             x = th.LongTensor(x)
#             x = F.one_hot(x,num_classes = 50257)
#             y = F.one_hot(y - th.ones_like(y),num_classes =4).float()
#             y_hat = test_model(x.float())
#             dot = torchviz.make_dot(y_hat.mean(), params=dict(test_model.named_parameters()), show_attrs=True, show_saved=True)
#             dot.view("model_graph.dot")
#             dot.render()
#             break
#             loss = loss_fun(y_hat,y)
#             loss.backward()
#             optimizer.step()
#             print(f"iteration - {i}\tepoch - {epoch}\tloss - {loss}\r")

if __name__ == '__main__':
    experiment(10,32,.001)
    # viz_experiment()