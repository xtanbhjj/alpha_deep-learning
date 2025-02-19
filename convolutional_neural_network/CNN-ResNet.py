#! coding: utf-8

import sys
sys.path.append('../')
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
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
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
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
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.net(x)


def accuracy(x, y): 
    if len(x.shape) == 2 and x.shape[1] == 10:
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


def main():
    #hyper 
    batch_size, num_epochs, lr  = 128, 10, 0.1
    # Dataloader
    trans = [transforms.ToTensor()]
    trans.insert(0, transforms.Resize(size=96))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    #model
    loss = nn.CrossEntropyLoss()
    model = ResNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #device
    device = dlf.devices()[0]
    model = model.to(device)
    print(device)

    for i in range(num_epochs):
        lo, train_per = train(model, optimizer, loss, device, train_iter)
        test_per = test(model, loss, device, test_iter)
        print('Epoch [%d/%d], Loss: %.4f' % (i + 1, num_epochs, lo))
        print('Train_curr: %.4f Test_curr: %.4f' % (train_per, test_per))


if __name__ == "__main__":
    main()