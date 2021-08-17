#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:09:59 2021

@author: ziqi
"""
import torch
import torchvision
import torch.nn as nn

class Triangular(nn.Module):
    def __init__(self):
         super(Triangular, self).__init__()
    def forward(self, input):
        out = nn.functional.relu(input + 1) - 2 * nn.functional.relu(input) + nn.functional.relu(input - 1)
        return out

def convert_relu_to_triangular(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Triangular())
        else:
            convert_relu_to_triangular(child)
            
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

class marco_softmax(nn.Module):
    def __init__(self, num_classes):
        super(marco_softmax, self).__init__()
        self.num_classes = num_classes
        self.e = torch.eye(num_classes).cuda()
    def forward(self, input):        
        nu = []
        pos = []
        for i in range(self.num_classes):
            nu.append(1/((input-self.e[i])**2).sum(1) + 1e-20)
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
    def __init__(self, network, num_classes, conservative, a, triangular):
        super(Net, self).__init__()
        if network=='vgg16':
            self.net = torchvision.models.vgg16(num_classes = num_classes, pretrained=False)
        elif network=='vgg19':
            self.net = torchvision.models.vgg19(pretrained=True)
            self.net.classifier[6] = nn.Linear(4096,num_classes)
        elif network=='resnet18':
            self.net = torchvision.models.resnet18(num_classes = num_classes, pretrained=False)
        elif network=='resnet50':
            self.net = torchvision.models.resnet50(pretrained=True)
            self.net.fc = nn.Linear(2048, num_classes)
        elif network=='squeezenet':
            self.net = torchvision.models.squeezenet1_0(num_classes = num_classes, pretrained=False)

        self.conservative = conservative
        if triangular:
            convert_relu_to_triangular(self.net)
        if conservative == 'double':
            self.softmax = nn.Softmax()
        elif conservative == 'monotone':
            self.softmax = conservative_softmax_monotone(num_classes, a)
        elif conservative == 'center':
            self.softmax = conservative_softmax(num_classes, a)
        elif conservative == 'marco':
            self.softmax = marco_softmax(num_classes)

    def forward(self, x):
        z = self.net(x)
        if self.conservative == 'False':
            return z, z
        else:
            x = self.softmax(z)
            return x, z
