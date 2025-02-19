#! coding: utf-8

import sys
sys.path.append('../')
import torch
import os
import torchvision
import torch.nn as nn
import torch.utils.data as Data 
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
import pandas as pd
from utils.accumulator import Accumulator 
import utils.dlf as dlf

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv=False, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        if use_conv:
            self.bypass = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.bypass = None
        
        self.end_relu = nn.ReLU()
    
    def forward(self, x):
        y = self.layer(x)
        if self.bypass:
            x = self.bypass(x)
        
        y += x
        return self.end_relu(y)
    
class ResNet_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual, init_block=False):
        super(ResNet_block, self).__init__()
        layer = []
        for i in range(num_residual):
            if i == 0 :
                if not init_block:
                    layer.append(Residual(in_channels, out_channels, True, 2))
                else:
                    layer.append(Residual(in_channels, out_channels)) 
            else:
                layer.append(Residual(out_channels, out_channels))
        
        self.layer = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.layer(x)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.init_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.net = nn.Sequential(
            self.init_block,
            ResNet_block(64, 64, 2, True),
            ResNet_block(64, 128, 2),
            ResNet_block(128, 256, 2),
            ResNet_block(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 176)
        )
    
    def forward(self, x):
        return self.net(x)
    
'''
Dataset 是一个抽象类,可以与DataLoader配合使用得到dataloader。
所以我们其实在数据处理部分得到Dataset对象即可。

在复用Dataset时,我们需要分别实现
__len__:返回数据集的大小(样本数量)
__getitem__:根据索引返回单个样本(数据和标签)
'''

class LeavesSet(Data.Dataset):

    def __init__(self, images_path, images_label, train=True):
        self.imgs = [os.path.join('../data/classify-leaves/', image_path) for image_path in images_path] 
        if train:
            self.train = True
            self.labels = images_label
        else:
            self.train = False
        
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                ])
        
    def __getitem__(self,index):
        image_path = self.imgs[index]
        pil_img = Image.open(image_path)
        transform = self.transform
        data = transform(pil_img)

        if self.train:
            label = self.labels[index]
            return data, label
        else:
            return data 
    
    def __len__(self):
        return len(self.imgs)

def accuracy(x, y): 
    if len(x.shape) == 2 and x.shape[1] == 176:
        x = x.argmax(axis=1)
    cmp = x.type(y.dtype) == y
    return float(cmp.to(y.dtype).sum())

def train(model, optimizer, loss, device, train_iter):
    model.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()

        with torch.no_grad():
            metric.add(l * y.numel(), accuracy(y_hat, y), y.numel())
        
    return metric[0] / metric[2], metric[1] / metric[2]

def test(model, loss, device, test_iter):
    model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in test_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            l = loss(y_hat, y)
            metric.add(accuracy(y_hat, y), y.numel())

    return metric[0] / metric[1]

def infer(test_data, test_dataset, model, batch_size, device, labelencoder):
    model.eval()
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pred = []
    with torch.no_grad():
        for x in test_iter:
            x = x.to(device)
            y = model(x)
            y = y.argmax(axis=1)
            pred.extend(y.cpu().numpy().tolist())
    
    #print(pred)
    ans = labelencoder.inverse_transform(pred)
    test_data['label'] = pd.Series(ans)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv('submission.csv', index=False)

def main():
    #hyper 
    #batch_size, num_epochs, lr, k  = 128, 10, 0.1, 5    0.83
    batch_size, num_epochs, lr, k  = 64, 20, 0.05, 5

    # Dataloader
    train_data = pd.read_csv('../data/classify-leaves/train.csv')
    test_data = pd.read_csv('../data/classify-leaves/test.csv')

    # encode the train label
    labelencoder = LabelEncoder()
    labelencoder.fit(train_data['label'])
    train_data['label'] = labelencoder.transform(train_data['label'])

    '''
    fit()：学习数据中所有类别。
    transform()：将类别标签转换为数字标签。
    fit_transform():fit() 和 transform() 的合并，用于训练并转换数据。
    inverse_transform()：将编码的标签还原为原始类别标签。
    '''

    train_dataSet = LeavesSet(train_data['image'], train_data['label'], train=True)
    test_dataSet = LeavesSet(test_data['image'], images_label=0, train=False)

    #model
    loss = nn.CrossEntropyLoss()
    model = ResNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    kfold = KFold(n_splits=k, shuffle=True)

    '''
    # 首先创建KFold实例
    kfold = KFold(n_splits=k, shuffle=True)  

    # 这里得到的train_ids和valid_ids指的是索引,并不是分割后的数据集
    for train_ids, valid_ids in kfold.split(train_dataSet):
    
        #train_subset 和 valid_subset是根据索引分割出来的数据集
        train_subset = torch.utils.data.Subset(train_dataSet, train_ids)
        valid_subset = torch.utils.data.Subset(train_dataSet, valid_ids)
        train_iter = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(valid_subset, batch_size=batch_size, shuffle=False) 
    '''

    #device
    device = dlf.devices()[0]
    model = model.to(device)
    print(device)

    '''
    train_iter = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True)
    flag = True
    for x, y in train_iter:
        if flag:
            print(x.shape, y.shape)
            flag = False
        else:
            break 
    '''
    
    for train_ids, valid_ids in kfold.split(train_dataSet):
        #Get part-set
        train_subset = torch.utils.data.Subset(train_dataSet, train_ids)
        valid_subset = torch.utils.data.Subset(train_dataSet, valid_ids)
        train_iter = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

        for i in range(num_epochs):
            lo, train_per = train(model, optimizer, loss, device, train_iter)
            test_per = test(model, loss, device, test_iter)
            print('Epoch [%d/%d], Loss: %.4f' % (i + 1, num_epochs, lo))
            print('Train_curr: %.4f Test_curr: %.4f' % (train_per, test_per))

    infer(test_data, test_dataSet, model, batch_size, device, labelencoder)

if __name__ == "__main__":
    main()