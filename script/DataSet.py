from math import floor
from torchvision.transforms import transforms
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import normalize

class MyDataSet(object):
    def __init__(self, x_file, y_file):
        self.trnX = pd.read_pickle(x_file)
        self.trnY = pd.read_pickle(y_file)

    def __getitem__(self, index):
        data = self.trnX[index]
        data = torch.Tensor(data)
        data = data.to(torch.float32)
        # data = normalize(data, p=1, dim=1)

        label = self.trnY[index]
        # label = torch.Tensor([label])

        return data, label

    def __len__(self):
        return self.trnX.shape[0]


class TwoDataSet(object):
    def __init__(self, x_file, y_file):
        self.trnX = pd.read_pickle(x_file)
        self.trnY = pd.read_pickle(y_file)

    def __getitem__(self, index):
        data = self.trnX[index]
        data = torch.Tensor(data)
        data = data.to(torch.float32)

        label = self.trnY[index]
        label = torch.Tensor(label)

        return data, label[0], label[1]

    def __len__(self):
        return self.trnX.shape[0]


def load_my_dataset(args, x_file, y_file):
    data_set = MyDataSet(x_file, y_file)
    len = data_set.__len__()
    train_len = floor(len * 0.8)
    test_len = len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(data_set, [train_len, test_len])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle=True,
    )

    return train_loader, test_loader

def load_two_dataset(args, x_file, y_file):
    data_set = TwoDataSet(x_file, y_file)
    len = data_set.__len__()
    train_len = floor(len * 0.8)
    test_len = len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(data_set, [train_len, test_len])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle=True,
    )

    return train_loader, test_loader


class SpecDataset(object):
    def __init__(self, total_data, total_label, total_specialty, pred_specialty):
        self.total_data = total_data
        self.total_label = total_label
        self.total_specialty = total_specialty
        self.pred_specialty = pred_specialty

    def __getitem__(self, index):
        data = self.total_data[index]
        data = torch.Tensor(data)
        data = data.to(torch.float32)

        label = self.total_label[index]
        label = torch.Tensor([label])

        specialty = self.total_specialty[index]
        specialty = torch.Tensor([specialty])

        pred_spec = self.pred_specialty[index]
        pred_spec = torch.Tensor([pred_spec])

        return data, label, specialty, pred_spec

    def __len__(self):
        return len(self.total_data)

def load_spec_dataset(args, total_data, total_label, total_specialty, pred_specialty):
    data_set = SpecDataset(total_data, total_label, total_specialty, pred_specialty)
    len = data_set.__len__()
    train_len = floor(len * 0.8)
    test_len = len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(data_set, [train_len, test_len])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle=True,
    )

    return train_loader, test_loader

class ThreeDataset(object):
    def __init__(self, total_data, total_label, pred_specialty):
        self.total_data = total_data
        self.total_label = total_label
        self.pred_specialty = pred_specialty

    def __getitem__(self, index):
        data = self.total_data[index]
        data = torch.Tensor(data)
        data = data.to(torch.float32)

        label = self.total_label[index]

        pred_spec = self.pred_specialty[index]

        return data, label, pred_spec

    def __len__(self):
        return len(self.total_data)

def load_three_dataset(args, total_data, total_label, pred_specialty):
    data_set = ThreeDataset(total_data, total_label, pred_specialty)
    len = data_set.__len__()
    train_len = floor(len * 0.8)
    test_len = len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(data_set, [train_len, test_len])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle=True,
    )

    return train_loader, test_loader