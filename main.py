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
    def __init__(self,in_filters, out_filters):
        super(SSNet, self).__init__()
        self.conv = ConvLayer(in_filters, out_filters, kernel_size = 7, stride = 1)
        
    def forward(self, x):
        out = self.conv(x)
        return out

class SSNetMultiple(torch.nn.Module):
    def __init__(self):
        super(SSNetMultiple, self).__init__()
        self.children = []
        for cnt in range(8):
            if cnt == 0:
                in_filters, out_filters = 1,32
            else:
                in_filters, out_filters = 32,32
            self.children.append(SSNet(in_filters, out_filters))
        
        self.main = nn.Sequential(*self.children)
        
    def forward(self, x, queue = 1):
        outs = [x]
        for cnt,child in enumerate(self.main):
            if cnt<queue:
                outs.append(child(outs[-1]))
        return outs[-1]
    
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
model = SSNetMultiple()
lr = 0.1
optimizer = optim.SGD(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.02)
lossfunc = nn.MSELoss()
for epoch in range(8):
    if epoch>0:
        for cc,param in enumerate(model.main[epoch-1].parameters()):
            print(epoch-1,"grad is deactivated")
            param.requires_grad = False
    for cnt,sample in enumerate(dataset):
        if cnt<100:
            optimizer.zero_grad()
            image, label = sample
            out = model(image.unsqueeze(0), queue = epoch+1)
            sim_vec = sim_func(out)
            loss = lossfunc(sim_vec, torch.zeros(sim_vec.shape))
            loss_obs = torch.max(torch.abs(sim_vec-torch.zeros(sim_vec.shape)))
            print("Epoch: {}\tSample: {}\tLoss: {}\tLR: {}".format(epoch,cnt,loss_obs,optimizer.param_groups[0]["lr"]))
            loss.backward()
            optimizer.step()
    #scheduler.step()
    
weights = model.conv.conv2d.weight.data.numpy()
for cnt,weight in enumerate(weights):
    plt.figure()
    plt.imshow(weight[0])