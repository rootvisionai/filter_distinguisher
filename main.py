# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:38:55 2020

@author: evrim
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import itertools
import numpy as np
import cv2
from PIL import Image

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class SSNet(torch.nn.Module):
    def __init__(self,in_filters, out_filters):
        super(SSNet, self).__init__()
        self.conv1 = ConvLayer(in_filters, 64, kernel_size = 5, stride = 1)
        self.conv2 = ConvLayer(64, out_filters, kernel_size = 1, stride = 1)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        out = self.pool(self.conv2(self.relu(self.conv1(x))))
        return out

class SSNetMultiple(torch.nn.Module):
    def __init__(self,levels = 5):
        super(SSNetMultiple, self).__init__()
        self.children_ = []
        for cnt in range(levels):
            if cnt == 0:
                in_filters, out_filters = 3,16
            elif cnt == levels-1:
                in_filters, out_filters = 16,16
            else:
                in_filters, out_filters = 16,16
            self.children_.append(SSNet(in_filters, out_filters))
        
        self.main = nn.Sequential(*self.children_)
        
    def forward(self, x, queue = 1):
        outs = [x]
        for cnt,child in enumerate(self.main):
            if cnt<queue:
                outs.append(child(outs[-1]))
        return outs[-1]

def normalize(vector):
    norm = vector.norm(p=2, dim=0, keepdim=True)
    vector_normalized = vector.div(norm.expand_as(vector))
    return vector_normalized

def sim_func(layers):
    combinations = list(itertools.combinations(np.arange(0,layers.shape[1]), 2))
    similarity_vector = torch.empty(len(combinations))
    for cnt,comb in enumerate(combinations):
        first = layers[0][comb[0]].flatten()
        second = layers[0][comb[1]].flatten()
        first_norm = normalize(first)
        second_norm = normalize(second)
        similarity_vector[cnt] = torch.matmul(first_norm,second_norm.T)
    return similarity_vector

def cam_to_tensor(cam):
    if cam.isOpened():
        ret, frame_ = cam.read()
    else:
        cam.release()
        cam = cv2.VideoCapture(video_source)
        ret, frame_ = cam.read()
    frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame)
    image = transform(frame_pil)
    return image, frame_, cam # image - pytorch tensor, image - opencv array, opencv capture object

transform=transforms.Compose([
                            transforms.CenterCrop((360,360)),
                            transforms.Resize((224,224)),
                            transforms.ToTensor()
                            ])
#dataset = datasets.MNIST('../data',
#                         train=True,
#                         download=True,
#                         transform=transform)
model = SSNetMultiple(levels = 4)
try:
    model.load_state_dict(torch.load("./model_videoplayback_2.pth"))
    train = False
except:
    train = True
    model.train()
    
lr = 0.02
optimizer = optim.SGD(model.parameters(), lr=lr)
lossfunc = nn.MSELoss()

video_source = "./videoplayback_1.mp4"
cam = cv2.VideoCapture(video_source)

loss_obs = 0
epoch = 0
if train:
    while epoch<4:
#        if epoch>0:
#            for cc,param in enumerate(model.main[epoch-1].parameters()):
#                print(epoch-1,"grad is deactivated")
#                param.requires_grad = True
        for cnt in range(0,120000):
            image, _, cam = cam_to_tensor(cam) # get image tensor and capture object
            
            optimizer.zero_grad()
            out = model(image.unsqueeze(0), queue = epoch+1)
            sim_vec = sim_func(out)
            loss = lossfunc(sim_vec, torch.zeros(sim_vec.shape))
            loss_obs_ = torch.max(torch.abs(sim_vec-torch.zeros(sim_vec.shape)))
            loss_obs += loss_obs_
            loss.backward()
            optimizer.step()
            print("Epoch: {}\tSample: {}\tLoss: {}\tLR: {}".format(epoch,cnt,loss_obs_,optimizer.param_groups[0]["lr"]))
    
            if cnt%20 == 0 and cnt!=0:
                loss_obs = loss_obs/20
                TH = 0.3 if epoch<3 else 0.2
                print("Epoch: {}\tSample: {}\tLoss: {}\tLR: {}".format(epoch,cnt,loss_obs,optimizer.param_groups[0]["lr"]))
                if loss_obs<TH or cnt>7000:
                    epoch += 1
                    break
                loss_obs = 0

    torch.save(model.state_dict(), "./model_videoplayback_2.pth")

def generate_embedding(model,cam,queue = 3):
    
    # model: model that is usd to extract embedding
    # cam: opencv capture object
    # queue: level of model that we extract the embedding, final level is suggested
    
    image, frame, _ = cam_to_tensor(cam) # get image tensor and frame array
    embedding = model(image.unsqueeze(0), queue = queue).flatten()
    return embedding, frame

def compare_samples(e1,e2):
    
    # e1: e2: embedding
    
    first_norm = normalize(e1.flatten())
    second_norm = normalize(e2.flatten())
    return torch.matmul(first_norm,second_norm.T).detach().numpy()

def custom_center_crop_and_resize(frame, size_crop = 360, size_resize = 720):
    
    # frame: frame that is captured from opencv capture object
    # size: 
    
    midr, midc = int(frame.shape[0]/2), int(frame.shape[1]/2)
    frame_croped = frame[int(midr-(size_crop/2)): int(midr+(size_crop/2)),
                         int(midc-(size_crop/2)): int(midc+(size_crop/2)),:]
    frame_croped_resized = cv2.resize(frame_croped, (size_resize,size_resize), interpolation = cv2.INTER_AREA)
    return frame_croped_resized

embedding_list = []
def compare_continuous(model,
                       cam,queue, best_of = 64,
                       memory_size = 2048):
    
    # model: model that is usd to extract embedding
    # cam: opencv capture object
    # queue: level of model that we extract the embedding, final level is suggested
    # anchor_frame_change_interval: interval to update anchor image, unit: frame rate
        
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText_1 = (1,12)
    bottomLeftCornerOfText_2 = (1,27)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 1
    
    global embedding_list
    cnt_f = 0
    cnt_w = 0
    sim = 0
    while True:
        if sim<0.4:
            e1, f1 = generate_embedding(model,cam,queue = queue)
            #f1 = custom_center_crop_and_resize(f1,360)
            cv2.imshow('frame 1', f1)
        
        e2, f2 = generate_embedding(model,cam,queue = queue)
        embedding_list.append(e2.detach().numpy())
        if memory_size != -1:
            embedding_list_ = embedding_list[-memory_size:]
        embedding_list_np = np.array(embedding_list_)
        std = np.std(embedding_list_np, axis=0)
        
        pca_idx = std.argsort()[-best_of:][::-1]
        e1_pca = e1[pca_idx.tolist()]
        e2_pca = e2[pca_idx.tolist()]
        
        sim = compare_samples(e1_pca,e2_pca)
        
        #f2 = custom_center_crop_and_resize(f2,360)
        
        zeros = np.zeros(e2.shape)
        zeros[pca_idx.tolist()] = 1
        zeros = zeros.reshape(16,10,10)
        zeros = np.sum(zeros.reshape(16,10,10),axis=0)
        irows, icolumns = np.where(zeros>=1)
        values = zeros[np.where(zeros>=1)]
        coordinates = [elm for elm in zip(irows/10, icolumns/10, values)]
        for elm in coordinates:
            cv2.circle(f2,(int(elm[0]*360+155),int(elm[1]*360+15)),int(elm[2]*3),(255,255,255),1)
        
        cv2.rectangle(f2, (0, 0), (130,30), (64,64,64), -1)
        
        cv2.putText(f2,'Similarity: {}'.format(str(np.round(sim, 3))), 
            bottomLeftCornerOfText_1, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(f2,'Frame: {}'.format(cnt_w), 
            bottomLeftCornerOfText_2, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
        cv2.imshow('frame 2', f2)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        cnt_f += 1
        cnt_w += 1
    
compare_continuous(model,cam,queue = 5,
                   memory_size = 512,
                   best_of = 64)