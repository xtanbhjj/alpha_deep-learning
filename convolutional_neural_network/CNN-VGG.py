#! coding: utf-8

'''
多个vgg_block + 全连接层
block 的使用，有效的设计复杂的网络
'''
import sys
sys.path.append('../')
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.accumulator import Accumulator 
import utils.dlf as dlf
    
class VGG_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGG_block, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layer.append(nn.ReLU())
            in_channels = out_channels
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)

class VGG(nn.Module):
    def __init__(self, conv_arch):
        super(VGG, self).__init__()
        in_channels = 1
        layer = []
        for (num_convs, out_channels) in conv_arch:
            layer.append(VGG_block(in_channels, out_channels, num_convs))
            in_channels = out_channels
        
        layer.append(nn.Flatten())
        layer.append(nn.Linear(out_channels * 7 * 7, 4096))
        layer.append(nn.ReLU())
        layer.append(nn.Dropout(0.5))
        layer.append(nn.Linear(4096, 4096))
        layer.append(nn.ReLU())
        layer.append(nn.Dropout(0.5))
        layer.append(nn.Linear(4096, 10))
        self.net = nn.Sequential(*layer)
    
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
    batch_size, num_epochs, lr  = 128, 10, 0.05
    # Dataloader
    trans = [transforms.ToTensor()]
    trans.insert(0, transforms.Resize(size=224))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    #model
    loss = nn.CrossEntropyLoss()
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = VGG(conv_arch)
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
