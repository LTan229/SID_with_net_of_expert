import torch
import torch.nn as nn
from torchvision import transforms as T
import pandas as pd
import numpy as np
import visualization as v
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
    if torch.cuda.is_available():
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
    train_correct += predict.eq(train_ls.view(-1,1)).sum().item
    train_acc += train_correct / train_ds.shape[0]

    print('Train loss: {:.3f}'.format(train_loss.item()))
    print('Train acc: {:.3f}%'.format(train_acc*100))

    return train_loss.item(),train_acc

def for_test(test_ds, test_ls,model,criterion):
    test_ds = test_ds.cuda()
    test_ls = test_ls.cuda()
    model = model.eval()
    correct = 0

    with torch.no_grad():
        output = model(test_ds)
        test_loss = criterion(output,test_ls)

        predict = output.max(1,keepdim=True)[1]
        correct += predict.eq(test_ls.view_as(predict)).sum().item()
        acc = correct / test_ds.shape[0]

    print('Test loss: {:.3f}'.format(test_loss.item()))
    print('Test acc: {:.3f}%'.format(acc * 100))
    return test_loss.item(), acc

def cross_validation(k, c_datas, c_labels, c_model, c_epochs, c_criterion, c_opt, result_name):
    examples = c_datas.shape[0]
    batch_examples = examples // k
    performs = []
    t_loss_list = []
    t_acc_list = []
    v_loss_list = []
    v_acc_list = []
    start_time = time.perf_counter()
    for i in range(k):
        print("第{}次训练和测试结果：".format(i + 1))
        if i == k - 1:
            test_datas = c_datas[i * batch_examples:, :]
            test_labels = c_labels[i * batch_examples:]
        else:
            test_datas = c_datas[i * batch_examples:(i + 1) * batch_examples, :]
            test_labels = c_labels[i * batch_examples:(i + 1) * batch_examples]
        train_datas = torch.cat((c_datas[0:i * batch_examples, :], c_datas[(i + 1) * batch_examples:, :]), dim=0)
        train_labels = torch.cat((c_labels[0:i * batch_examples], c_labels[(i + 1) * batch_examples:]), dim=0)
        t_loss, t_acc = train(train_datas, train_labels, c_model, c_criterion, c_opt, epochs=c_epochs)
        v_loss, v_acc = for_test(test_datas, test_labels, c_model, c_criterion)
        torch.cuda.empty_cache()
        performs.append((v_acc, v_loss))

        t_loss_list.append(t_loss)
        t_acc_list.append(t_acc)
        v_loss_list.append(v_loss)
        v_acc_list.append(v_acc)

    end_time = time.perf_counter()
    performs = sorted(performs)
    print("{}次中最佳性能 --> loss: {:.3f}, accuracy: {:.2f}%".format(k, performs[-1][1], performs[-1][0] * 100))
    print("{}次中最差性能 --> loss: {:.3f}, accuracy: {:.2f}%".format(k, performs[0][1], performs[0][0] * 100))

    t_loss_avg = np.mean(np.array(t_loss_list))
    t_acc_avg = np.mean(np.array(t_acc_list))
    v_loss_avg = np.mean(np.array(v_loss_list))
    v_acc_avg = np.mean(np.array(v_acc_list))
    print("{}次平均性能:"
        "\n\t--> Train loss: {:.3f}, Train accuracy: {:.2f}%"
        "\n\t--> Test loss: {:.3f}, Test accuracy: {:.2f}%"
        .format(k, t_loss_avg, t_acc_avg * 100, v_loss_avg, v_acc_avg * 100))

    t_loss_var = np.var(np.array(t_loss_list))
    t_acc_var = np.var(np.array(t_acc_list))
    v_loss_var = np.var(np.array(v_loss_list))
    v_acc_var = np.var(np.array(v_acc_list))
    print("{}次结果方差:"
    "\n\t--> Train loss: {:.8f}, Train accuracy: {:.8f}"
    "\n\t--> Test loss: {:.8f}, Test accuracy: {:.8f}"
    .format(k, t_loss_var, t_acc_var, v_loss_var, v_acc_var))

    print("共耗时：{:.3f} s".format(end_time - start_time))
    v.visualization(t_loss_list, t_acc_list, v_loss_list, v_acc_list, result_name, is_save=True)