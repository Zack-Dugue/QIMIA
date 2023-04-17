import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torchtext as tt
from container import LearnedQueryAttention , LearnedQueryAttention2
from NLP import make_nlp_QIMIA,make_nlp_QIMIA2, make_positional_encoding
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer , AutoTokenizer,GPT2Tokenizer

class TestModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(50257, 256)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,dropout=0,batch_first=True,norm_first=True),6,norm =nn.LayerNorm(256))
        self.output_norm = nn.LayerNorm(256)
        self.FF1 = nn.Linear(256, 2048)
        self.Act1 = nn.GLU()
        self.FF2 = nn.Linear(1024, 2048)
        self.Act2 = nn.GLU()
        self.FF3 = nn.Linear(1024, 4)
        self.PE = make_positional_encoding(256)
    def forward(self,x):
        x = self.embed(x)
        PE = th.permute(self.PE[:x.size(1)],[1,0,2]).repeat([x.size(0), 1, 1])
        x += PE
        h = self.transformer(x)
        xdims = list(x.size())
        h = h.view(xdims[0],xdims[1],256)
        h = th.squeeze(F.avg_pool1d(th.permute(h,[0,2,1]), kernel_size=xdims[1], stride=1))
        h = self.output_norm(h)
        h = self.FF1(h)
        h = self.Act1(h)
        h = self.FF2(h)
        h = self.Act2(h)
        h = self.FF3(h)
        return h

class TestModel(nn.Module):
    def __init__(self,embed_dim,key_dim,num_layers):
        super(TestModel, self).__init__()
        self.embed_dim = embed_dim
        self.base = make_nlp_QIMIA2(embed_dim,key_dim,num_layers)
        self.output_attention = LearnedQueryAttention(key_dim,8,v_dim=embed_dim,w0=True)
        # self.global_mean = nn.AdaptiveAvgPool1d(1)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.FF1 = nn.Linear(embed_dim,2048)
        self.Act1 = nn.GLU()
        self.FF2 = nn.Linear(1024,2048)
        self.Act2 = nn.GLU()
        self.FF3 = nn.Linear(1024,4)
        self.input_key_embed = nn.Linear(50257, key_dim)
        self.input_val_embed = nn.Linear(50257, embed_dim)
        self.PE_key_encode = nn.Linear(embed_dim,embed_dim)
        self.PE_val_encode = nn.Linear(embed_dim,embed_dim)
        PE = make_positional_encoding(embed_dim)


    def forward(self,x):
        #idk why
        x_key = self.input_key_embed(self.base.flatten_to_token(x))
        x_val = self.input_val_embed(self.base.flatten_to_token(x))
        PE = make_positional_encoding(self.embed_dim)[:x.size(1)]
        PE_key = self.base.flatten_to_token(self.PE_key_encode(PE).repeat([x.size(0),1,1]))
        PE_val = self.base.flatten_to_token(self.PE_val_encode(PE).repeat([x.size(0),1,1]))
        keys = th.stack([x_key,PE_key],1)
        values = th.stack([x_val,PE_val],1)

        keys,values = self.base(keys,values, [x.size(1),self.embed_dim])
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
    # test_model = TestModel(256,256,6)
    test_model = TestModel2()
    loss_fun = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(test_model.parameters(),lr)
    for epoch in range(epochs):
        i = 0
        for y,x in dataloader:
            i += 1
            optimizer.zero_grad()
            x = auto_tokenize(x,padding=True)['input_ids']
            x = th.LongTensor(x)
            x = F.one_hot(x,num_classes = 50257)
            y = F.one_hot(y - th.ones_like(y),num_classes =4).float()
            y_hat = test_model(x.float())
            loss = loss_fun(y_hat,y)
            loss.backward()
            optimizer.step()
            print(f"i - {i} \t epoch - {epoch} \t loss - {loss}\r")

if __name__ == '__main__':
    experiment(10,32,.001)
