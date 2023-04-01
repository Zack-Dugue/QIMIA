import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from container import LearnedQueryAttention , LearnedQueryAttention2
from NLP import make_nlp_QIMIA
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer , AutoTokenizer,GPT2Tokenizer


class TestModel(nn.Module):
    def __init__(self,embed_dim,key_dim,num_layers):
        super(TestModel, self).__init__()
        self.embed_dim = embed_dim
        self.base = make_nlp_QIMIA(50257,embed_dim,key_dim,num_layers)
        self.output_attention = LearnedQueryAttention(key_dim,8,v_dim=embed_dim,w0=True)
        # self.global_mean = nn.AdaptiveAvgPool1d(1)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.FF1 = nn.Linear(embed_dim,2048)
        self.Act1 = nn.GLU()
        self.FF2 = nn.Linear(1024,2048)
        self.Act2 = nn.GLU()
        self.FF3 = nn.Linear(1024,4)

    def forward(self,x):
        keys,values = self.base(x)
        h = self.output_attention((keys,values))
        xdims = list(x.size())
        h = h.view(xdims[0],xdims[1],self.embed_dim)
        h = th.squeeze(F.avg_pool1d(th.permute(h,[0,2,1]), kernel_size=xdims[1], stride=1))
        h = self.output_norm(h)
        h = self.FF1(h)
        h = self.Act1(h)
        h = self.FF2(h)
        h = self.Act2(h)
        h = self.FF3(h)
        return h

def experiment(epochs,batch_size,lr):

    auto_tokenize = GPT2Tokenizer.from_pretrained("gpt2")
    auto_tokenize.pad_token = auto_tokenize.eos_token
    dataset = tt.datasets.AG_NEWS()[0]
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    test_model = TestModel(256,256,6)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(test_model.parameters(),lr)
    for epoch in range(epochs):
        for y,x in dataloader:
            optimizer.zero_grad()
            x = auto_tokenize(x,padding=True)['input_ids']
            x = th.LongTensor(x)
            x = F.one_hot(x,num_classes = 50257)
            y = F.one_hot(y - th.ones_like(y),num_classes =4).float()
            y_hat = test_model(x.float())
            loss = loss_fun(y_hat,y)
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch} \t loss - {loss}\r")

if __name__ == '__main__':
    experiment(10,32,.001)
