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

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.network = nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.network(x)

def accuracy(x, y): #计算预测正确的数目
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
    # Dataloader
    batch_size, num_epochs, lr  = 256, 20, 0.9
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=False)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    #model
    loss = nn.CrossEntropyLoss()
    model = LeNet()
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
