import torch
import torch.nn as nn
from torchvision import transforms as T
import pandas as pd
import numpy as np
# import Visualization as v
import time




class Net(nn.Module):
    def __init__(self,input, output):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input,output),
            nn.Tanh()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(output, output),
            nn.Tanh()
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(output, output),
            nn.Softmax()
        )
    def forward(self,x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)

def get_datas(file):
    original_data = pd.read_csv(file)
    datas_array = original_data.to_numpy()
    np.random.seed(5)
    np.random.shuffle(datas_array)

    ds = datas_array[:,:-1]
    ls = datas_array[:,-1]

    ds = torch.from_numpy(ds).float()
    ls = torch.from_numpy(ls).long()

def init(model):
    for m in model.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)

def train(train_ds,train_ls,model,criterion,optimizer,epochs=20):
    train_correct = 0
    train_acc = 0
    train_ds = train_ds.cuda()
    train_ls = train_ls.cuda()
    model = model.cuda()
    init(model)
    for epochs in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_ds)
        train_loss = criterion(output,train_ls)
        train_loss.backward()
        optimizer.step()
    predict = output.max(1,keepdim=True)[1]
    train_correct += predict.eq(train_ls.view(predict)).sum().item
    train_acc += train_correct / train_ds.shape[0]

    print('Train loss: {:.3f}'.format(train_loss.item()))
    print('Train acc: {:.3f}%'.format(train_acc*100))

    return train_loss.item(),train_acc

def for_pred(pred_ds, model):
    pred_ds = pred_ds.cuda()
    model = model.eval()


    with torch.no_grad():
        output = model(pred_ds)

        predict = output.max(1,keepdim=True)[1]

    return output,predict

