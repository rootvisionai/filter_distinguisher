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
        #reflection_padding = kernel_size // 2
        #self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        #out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out

class SSNet(torch.nn.Module):
    def __init__(self,in_filters, out_filters):
        super(SSNet, self).__init__()
        self.conv1 = ConvLayer(in_filters, 64, kernel_size = 5, stride = 1)
        self.conv2 = ConvLayer(64, out_filters, kernel_size = 1, stride = 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        #out = self.conv1(x)
        out = self.conv2(self.relu(self.conv1(x)))
        return out

class SSNetMultiple(torch.nn.Module):
    def __init__(self):
        super(SSNetMultiple, self).__init__()
        self.children = []
        for cnt in range(4):
            if cnt == 0:
                in_filters, out_filters = 1,16
            elif cnt == 3:
                in_filters, out_filters = 16,4
            else:
                in_filters, out_filters = 16,16
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
        transforms.ToTensor()
        ])
dataset = datasets.MNIST('../data',
                         train=True,
                         download=True,
                         transform=transform)
model = SSNetMultiple()
lr = 0.8
optimizer = optim.SGD(model.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.02)
lossfunc = nn.MSELoss()
loss_obs = 0
epoch = 0
while epoch<4:
#    if epoch>0:
#        for cc,param in enumerate(model.main[epoch-1].parameters()):
#            print(epoch-1,"grad is deactivated")
#            param.requires_grad = True
    for cnt,sample in enumerate(dataset):
        optimizer.zero_grad()
        image, label = sample
        out = model(image.unsqueeze(0), queue = epoch+1)
        sim_vec = sim_func(out)
        loss = lossfunc(sim_vec, torch.zeros(sim_vec.shape))
        loss_obs_ = torch.max(torch.abs(sim_vec-torch.zeros(sim_vec.shape)))
        loss_obs += loss_obs_
        loss.backward()
        optimizer.step()
        print("__Loss: {}__".format(loss_obs_))

        if cnt%10 == 0 and cnt!=0:
            loss_obs = loss_obs/10
            print("Epoch: {}\tSample: {}\tLoss: {}\tLR: {}".format(epoch,cnt,loss_obs,optimizer.param_groups[0]["lr"]))
            if loss_obs<0.40:
                epoch += 1
                break
            loss_obs = 0

    #scheduler.step()
    
weights = model.main[-1].conv.conv2d.weight.data.numpy()
for cnt,weight in enumerate(weights):
    plt.figure()
    plt.imshow(weight[0])
    
for cnt,layer in enumerate(out[0]):
    plt.figure()
    plt.imshow(layer.detach())

def test(test_index = 0):
    img0,label0 = dataset[test_index]
    out0 = model(img0.unsqueeze(0), queue = 5)
    
    results = []
    for cnt,sample in enumerate(dataset):
        if cnt<10000 and cnt!=test_index:
            img1, label1 = sample
            out1 = model(img1.unsqueeze(0), queue = 5)
            
            first = out0.flatten()
            second = out1.flatten()
            
            first_norm = (first - torch.mean(first)) / (torch.std(first) * len(first))
            second_norm = (second - torch.mean(second)) / (torch.std(second))
            
            results.append(["{}-{}".format(label0,label1),torch.matmul(first_norm,second_norm.T).detach().numpy()])
        
    sorted_results = sorted(results,key = lambda x:x[1],reverse=True)
    print(sorted_results[0:10],"\n")
    return sorted_results

sr1 = test(test_index = 2000)
sr2 = test(test_index = 2001)
sr3 = test(test_index = 2002)
sr4 = test(test_index = 2003)
sr5 = test(test_index = 2004)