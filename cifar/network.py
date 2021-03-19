#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:09:59 2021

@author: ziqi
"""
import torch
import torchvision
import torch.nn as nn

class conservative_softmax(nn.Module): 
    def __init__(self, num_classes, a):
        super(conservative_softmax, self).__init__()
        self.num_classes = num_classes
        self.a = a
    def forward(self, input):
        nu = []
        pos = []
        for i in range(self.num_classes):
            nu.append(1/((self.a*input[:,i])**2 + 1e-20))
        for i in range(self.num_classes):
            pos.append(nu[i]/sum(nu))
        pos = torch.stack(pos, 1)
        return pos
    
    
class conservative_softmax_monotone(nn.Module): 
    def __init__(self, num_classes, a):
        super(conservative_softmax_monotone, self).__init__()
        self.num_classes = num_classes
        self.a = a
    def forward(self, input):
        nu = []
        pos = []
        for i in range(self.num_classes):
            nu.append(input[:,i] + torch.sqrt(1 + (input[:,i])**2))
        for i in range(self.num_classes):
            pos.append(nu[i]/sum(nu))
        pos = torch.stack(pos, 1)
        return pos
    


use_gpu = torch.cuda.is_available()

class Net(nn.Module):
    def __init__(self, num_classes, conservative, a):
        super(Net, self).__init__()
        self.vgg = torchvision.models.vgg16(num_classes = num_classes, pretrained=False)
        if not conservative:
            self.softmax = nn.Softmax()
        elif conservative == 'monotone':
            self.softmax = conservative_softmax_monotone(num_classes, a)
        elif conservative == 'center':
            self.softmax = conservative_softmax(num_classes, a)

    def forward(self, x):
        z = self.vgg(x)
        x = self.softmax(z)
        return x, z
