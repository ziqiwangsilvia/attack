#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:06:05 2021

@author: ziqi
"""

import torch
import torchvision
import torchvision.transforms as transforms

def prepare_dataset(train_all, train_index, test_all, test_index, mode):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    if not train_all:
        train_classes = {}
        new_label = 0        
        for i in train_index:
                train_classes[i] = new_label
                new_label = new_label + 1
                
    if mode == 'train':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        if not train_all: 
            idx = torch.ByteTensor(len(trainset.targets))*0
            for i in train_index:
                idx = idx | (trainset.targets==i)
            trainset.targets = trainset.targets[idx]
            trainset.data = trainset.data[idx]
            for k,v in train_classes.items():
                trainset.targets[trainset.targets==k] = v
        return trainset
    
    else:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        if not test_all:
            idx = torch.ByteTensor(len(testset.targets))*0
            for i in test_index:
                idx = idx | (testset.targets==i)
            testset.targets = testset.targets[idx]
            testset.data = testset.data[idx]
        if not train_all:
            for k,v in train_classes.items():
                testset.targets[testset.targets==k] = v        
        return testset
            


