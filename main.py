# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:38:55 2020

@author: evrim
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import itertools
import numpy as np
import matplotlib.pyplot as plt

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class SSNet(torch.nn.Module):
    def __init__(self):
        super(SSNet, self).__init__()
        self.conv = ConvLayer(1, 32, kernel_size = 7, stride = 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.conv(x))
        return out
    
def sim_func(layers):
    combinations = list(itertools.combinations(np.arange(0,layers.shape[1]), 2))
    similarity_vector = torch.empty(len(combinations))
    for cnt,comb in enumerate(combinations):
        first = layers[0][comb[0]].flatten()
        second = layers[0][comb[1]].flatten()
        first_norm = (first - torch.mean(first)) / (torch.std(first) * len(first))
        second_norm = (second - torch.mean(second)) / (torch.std(second))
        similarity_vector[cnt] = torch.matmul(first_norm,second_norm.T)
    return similarity_vector
    
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset = datasets.MNIST('../data',
                         train=True,
                         download=True,
                         transform=transform)
model = SSNet()
lr = 0.4
optimizer = optim.SGD(model.parameters(), lr=lr)
lossfunc = nn.MSELoss()
for cnt,sample in enumerate(dataset):
    optimizer.zero_grad()
    image, label = sample
    out = model(image.unsqueeze(0))
    sim_vec = sim_func(out)
    loss = lossfunc(sim_vec, torch.zeros(sim_vec.shape))
    loss.backward()
    optimizer.step()
    loss_obs = torch.max(torch.abs(sim_vec-torch.zeros(sim_vec.shape)))
    print(loss_obs)
    
weights = model.conv.conv2d.weight.data.numpy()
for cnt,weight in enumerate(weights):
    plt.figure()
    plt.imshow(weight[0])