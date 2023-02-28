import pickle
import torch
import numpy as np

with open('data/fft_result_wo_simplify/trnX.pkl', 'rb') as f:
  trnX = pickle.load(f)

with open('data/trnY_specialty.pkl', 'rb') as f:
# with open('data/fft_result_wo_simplify/trnY.pkl', 'rb') as f:
  trnY = pickle.load(f)

trnY = trnY[:,1]

# _range = np.max(trnX) - np.min(trnX)
# trnX = (trnX - np.min(trnX)) / _range
trnX = torch.Tensor(trnX)

trnY = torch.Tensor(trnY)
trnY = trnY.long()
print(trnY.shape)

import torch
import torch.nn as nn
from torchvision import transforms as T
import pandas as pd
import numpy as np
# import visualization as v
import time


from script.NeuralNetwork import NeuralNetwork
from script.train import train
from torch import nn
from torch.optim import Adam

net = NeuralNetwork(250)

train(trnX, trnY, net, nn.CrossEntropyLoss(), Adam(net.parameters(), weight_decay=1e-5, lr=1e-3), epochs=500)