import os
import unittest
from unittest.mock import patch
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision
import torchinfo
import sys
from pathlib import Path
from typing import Any
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.accumulator import Accumulator
from utils.plot import ImageUtils
from utils.timer import Timer
import utils.dlf as dlf

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
content_img = Image.open('cat.png')
style_img= Image.open('cartoon.png')
content_img = content_img.convert('RGB')

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

def extract_features(X, content_layers, style_layers, model):
    contents = []
    styles = []
    for i in range(len(model)):
        X = model[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(image_shape, device, model):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers, model)
    return content_X, contents_Y

def get_styles(image_shape, device, model):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers, model)
    return style_X, styles_Y

class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def train(contents_Y, styles_Y_gram, device, loss, model, optimizer, init_img):
    optimizer.zero_grad()
    content_Y_hat, style_Y_hat = extract_features(init_img, content_layers, style_layers, model)
    contents_l, styles_l, tv_l, l = loss(
            init_img, content_Y_hat, style_Y_hat, contents_Y, styles_Y_gram)
    l.backward()
    optimizer.step()
    return init_img

def main():
    content_weight, style_weight, tv_weight = 1, 1e3, 10
    lr, num_epochs, image_shape = 0.3, 500, (300, 450)
    # model
    pretrained_net = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)])
    device = dlf.devices('cpu')[0]
    model = model.to(device)
    #data
    content_X, contents_Y = get_contents(image_shape, device, model)
    _, styles_Y = get_styles(image_shape, device, model)
    init_img = SynthesizedImage(content_X.shape).to(device)
    init_img.weight.data.copy_(content_X.data)
    optimizer = torch.optim.Adam(init_img.parameters(), lr=lr)
    def gram(X):
        num_channels, n = X.shape[1], X.numel() // X.shape[1]
        X = X.reshape((num_channels, n))
        return torch.matmul(X, X.T) / (num_channels * n)
    def content_loss(Y_hat, Y):
        return torch.square(Y_hat - Y.detach()).mean()
    def style_loss(Y_hat, gram_Y):
        return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
    def tv_loss(Y_hat):
        return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                    torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
    def loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
        # 分别计算内容损失、风格损失和全变分损失
        contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
            contents_Y_hat, contents_Y)]
        styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
            styles_Y_hat, styles_Y_gram)]
        tv_l = tv_loss(X) * tv_weight
        # 对所有损失求和
        l = sum(10 * styles_l + contents_l + [tv_l])
        return contents_l, styles_l, tv_l, l

    styles_Y_gram = [gram(Y) for Y in styles_Y]
    init_img = init_img()
    '''
    1. Python调用init_img的__call__方法。
    2. __call__方法内部调用init_img的forward()方法。
    3. forward()方法返回self.weight,这个返回值被赋值给变量init_img。
    '''

    for i in range(num_epochs):
        init_img = train(contents_Y, styles_Y_gram, device, loss, model, optimizer, init_img)
        #sys.exit("successfully")
    img = postprocess(init_img)
    img.show()

if __name__ == '__main__':
    main()