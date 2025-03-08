#! coding: utf-8

import sys
sys.path.append('../')
import torch
import os
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as Data 
from torch.utils.data import DataLoader
import pandas as pd
from utils.accumulator import Accumulator 
import utils.dlf as dlf

dlf.DATA_HUB['voc2012'] = (dlf.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()

    def normalize_image(self, img):
        return self.transform(img.float())

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
  
def train(model, optimizer, loss, device, train_iter):
    model.train()
    metric = Accumulator(2)
    for x, y in train_iter:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        l = loss(y_hat, y)
        l = l.sum()
        l.backward()
        optimizer.step()
        # sys.exit("successfully")

        with torch.no_grad():
            metric.add(l * y.numel(), y.numel())
    
    return metric[0] / metric[1]

def test(model, loss, device, test_iter):
    model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in test_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            l = loss(y_hat, y)
            l = l.sum()
            metric.add(l * y.numel(), y.numel())
        
    return metric[0] / metric[1]

def tensor_image_show(img):
    plt.imshow(img.numpy())
    plt.show()
    print("图像形状:", img.numpy().shape) 

def inference(model, device, test_iter):
    model.eval()
    with torch.no_grad():
        for x, y in test_iter:
            x = x[0:1]
            x = x.to(device)
            pred = model(x).argmax(dim=1)
            pred.reshape(pred.shape[1], pred.shape[2])
            colormap = torch.tensor(VOC_COLORMAP, device=device)
            X = pred.long()
            X = X.squeeze(dim=0)
            X = colormap[X, :]
            x = x.squeeze(dim=0)
            x = x.permute(1, 2, 0)
            tensor_image_show(x)
            tensor_image_show(X)
            break

def main():
    batch_size, num_epochs, lr, wd, crop_size = 64, 10, 0.001, 1e-3, (320, 480)

    #DataSet
    voc_dir = dlf.download_extract('voc2012', 'VOCdevkit/VOC2012')
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size, shuffle=True, drop_last=True)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size, drop_last=True)

    #model
    pre_trained = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(pre_trained.children())[:-2])
    num_classes = 21
    model.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    model.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                    kernel_size=64, padding=16, stride=32))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    W = bilinear_kernel(num_classes, num_classes, 64)
    model.transpose_conv.weight.data.copy_(W)

    def loss(inputs, targets):
        return nn.functional.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
    
    #device
    device = dlf.devices('cpu')[0]
    model = model.to(device)
    print(device)

    for i in range(num_epochs):
        train_loss = train(model, optimizer, loss, device, train_iter)
        test_loss = test(model, loss, device, test_iter)
        print(train_loss, test_loss)

    #inference
    inference(model, device, test_iter)


if __name__ == '__main__':
    main()