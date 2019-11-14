import math
import numpy as np
import torch
from torch.nn.functional import kl_div
from torch.distributions.multinomial import Categorical
from torch import nn

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[],softmax=False):
        super(NN, self).__init__()
        self.softmax = softmax
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
            self.layers.append(nn.Linear(inSize, outSize))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.tanh(x)
            x = self.layers[i](x)
        return torch.softmax(x,0) if self.softmax else x

def phi(obs,device):
    return torch.Tensor(obs).to(device,torch.double)
