#! coding: utf-8
import sys
sys.path.append('../')
import torch
import os
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils.accumulator import Accumulator 
import utils.dlf as dlf
from exp.pre_data import *
from accelerate import Accelerator

accelerator = Accelerator()

def accuracy(x, y, threshold=0.5):
    """
    计算多标签分类的准确率，要求两个向量完全相等时才取 1。

    Args:
        x (torch.Tensor): 模型输出的 logits (形状为 [batch_size, num_labels]).
        y (torch.Tensor): 真实标签 (形状为 [batch_size, num_labels]，0 或 1).
        threshold (float): 将概率转换为二进制预测的阈值.

    Returns:
        float: 平均准确率 (只有预测向量和真实向量完全相等时才算正确).
    """
    probabilities = torch.sigmoid(x)
    predictions = (probabilities > threshold).float()
    correct_predictions = (predictions == y).all(dim=1).float()
    accuracy = correct_predictions.mean().item()
    return accuracy

def train(model, optimizer, loss, device, train_iter):
    model.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        x = x.to(device)
        y = y.to(device).float()
        optimizer.zero_grad()
        y_hat = model(x)
        l = loss(y_hat, y)
        accelerator.backward(l)
        optimizer.step()

        with torch.no_grad():
            metric.add(l * y.numel(), accuracy(y_hat, y) * y.numel(), y.numel())
    
    return metric[0] / metric[2], metric[1] / metric[2]

def test(model, loss, device, test_iter):
    model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in test_iter:
            x = x.to(device)
            y = y.to(device).float()
            y_hat = model(x)
            l = loss(y_hat, y)
            metric.add(accuracy(y_hat, y) * y.numel(), y.numel())

    return metric[0] / metric[1]

def main():
    batch_size, num_epochs, lr, k = 16, 20, 0.005, 5
    loss = nn.BCEWithLogitsLoss()
    train_dataSet = Get_data_conv()

    model = nn.Sequential()
    model.features = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.output = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 101))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)
    kfold = KFold(n_splits=k, shuffle=True)

    #device
    device = accelerator.device
    model = model.to(device)
    model, optimizer, train_dataSet = accelerator.prepare(model, optimizer, train_dataSet)
    print(device)
    
    #train
    for train_ids, valid_ids in kfold.split(train_dataSet):
        #Get part-set
        train_subset = torch.utils.data.Subset(train_dataSet, train_ids)
        valid_subset = torch.utils.data.Subset(train_dataSet, valid_ids)
        train_iter = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

        for i in range(num_epochs):
            l, train_per = train(model, optimizer, loss, device, train_iter)
            test_per = test(model, loss, device, test_iter)
            print('Epoch [%d/%d], Loss: %.4f' % (i + 1, num_epochs, l))
            print('Train_curr: %.4f Test_curr: %.4f' % (train_per, test_per))
    
    PATH = "../data/exp_data/conv_parameters.pth" 
    torch.save(model.state_dict(), PATH)
    
    
if __name__ == "__main__":
    main()