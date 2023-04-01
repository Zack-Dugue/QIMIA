import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import torch.nn.functional  as F
import torch
import json

model_dict = {}
data_dict = {}
optimizer_dict = {}
lr_scheduler_dict = {}



def parse_model(model_json_path):
    json.load(model_json_path)
    pass
def parse_data(data_json_path):
    pass
def parse_experiment(experiment_json_path):
    pass


if __name__ == '__main__':
    pass