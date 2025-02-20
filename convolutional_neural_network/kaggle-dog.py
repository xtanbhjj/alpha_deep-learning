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

class DogSet(Data.Dataset):

    def __init__(self, images_path, images_label, train=True):     
        if train:
            self.imgs = [os.path.join('../data/dog-breed-identification/train', image_path + '.jpg') for image_path in images_path]
            self.labels = images_label
            self.train = True
            self.trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.imgs = [os.path.join('../data/dog-breed-identification/test', image_path + '.jpg') for image_path in images_path]
            self.train = False
            self.trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]) 
        
    def __getitem__(self, index):
        image_path = self.imgs[index]
        pil_img = Image.open(image_path)
        data = self.trans(pil_img)

        if self.train:
            label = self.labels[index]
            return data, label
        else:
            return data
    
    def __len__(self):
        return len(self.imgs)

def accuracy(x, y): 
    if len(x.shape) == 2 and x.shape[1] == 120:
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
    id = test_data['id'].tolist()

    pred = []
    with torch.no_grad():
        for x in test_iter:
            x = x.to(device)
            y = model(x)
            y = torch.nn.functional.softmax(y, dim=1)
            pred.extend(y.cpu().detach().numpy())

    all_classes_in_order = labelencoder.classes_
    with open('submission.csv', 'w') as f:
        f.write('id,' + ','.join(all_classes_in_order) + '\n')
        for i, output in zip(id, pred):
            #print(i)
            f.write(i + ',' + ','.join(
                [str(num) for num in output]) + '\n')

def main():
    batch_size, num_epochs, lr, k = 128, 10, 0.005, 2

    # DataSet
    train_data = pd.read_csv('../data/dog-breed-identification/train.csv')
    test_data = pd.read_csv("../data/dog-breed-identification/test.csv")
    #print(test_data[0])

    labelencoder = LabelEncoder()
    labelencoder.fit(train_data['breed'])
    train_data['breed'] = labelencoder.transform(train_data['breed'])

    train_dataSet = DogSet(train_data['id'], train_data['breed'], train=True)
    test_dataSet = DogSet(test_data['id'], images_label=0, train=False)

    #model
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 120)
    loss = nn.CrossEntropyLoss()

    '''
    在微调的时候,我们提取特征的部分可以frozen,也就是说不更新梯度

    model = nn.Sequential()
    model.features = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.output = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 120))
    device = dlf.devices()[0]
    model = model.to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=lr)
    '''

    extract_para = [param for name, param in model.named_parameters()
                    if name not in ['fc.weight', 'fc.bias']]
    optimizer = torch.optim.SGD([{'params': extract_para},
                                 {'params': model.fc.parameters(), 'lr': lr * 10}],
                                 lr=lr)
    kfold = KFold(n_splits=k, shuffle=True)

    #device
    device = dlf.devices()[0]
    model = model.to(device)
    '''
    devices = [0, 1, 2, 3]
    model = nn.DataParallel(model, device_ids=devices)
    '''
    print(device)

    
    #train
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
        
