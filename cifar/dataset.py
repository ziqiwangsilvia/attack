#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:06:05 2021

@author: ziqi
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

def prepare_dataset(train_all, train_index, test_all, test_index, mode):
    transform_train = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


    
    if not train_all:
        train_classes = {}
        new_label = 0        
        for i in train_index:
                train_classes[i] = new_label
                new_label = new_label + 1
                
    if mode == 'train':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
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
                                       download=True, transform=transform_test)
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
    


def prepare_dataset_cifar100(train_all, train_index, test_all, test_index, mode):
    transform_train = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),])


    
    if not train_all:
        train_classes = {}
        new_label = 0        
        for i in train_index:
                train_classes[i] = new_label
                new_label = new_label + 1
                
    if mode == 'train':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
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
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
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
            


def load_imagenette(mode=None, transforms=None, ram_dataset=False):
    if mode == 'train':
        dataset = torchvision.datasets.ImageFolder(
            root='/tudelft.net/staff-bulk/ewi/insy/VisionLab/ziqiwang/teachingtolie/data/imagenette-320/train',
            transform=transforms)

    elif mode == 'val':
        dataset = torchvision.datasets.ImageFolder(
            root='/tudelft.net/staff-bulk/ewi/insy/VisionLab/ziqiwang/teachingtolie/data/imagenette-320/val',
            transform=transforms)
    return dataset


class Imagenette(data.Dataset):
    def __init__(self, mode=None, input_shape=None, ram_dataset=False):
        self.mode = mode
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.short_size = int(min(input_shape) / 0.875)
        self.input_shape = input_shape
        self.transform = transforms.Compose([
            transforms.Resize(self.short_size),
            transforms.CenterCrop(self.input_shape),
            transforms.ToTensor(),
            transforms.Normalize(*self.mean_std)
        ])

        self.data = load_imagenette(mode=self.mode, transforms=self.transform, ram_dataset=ram_dataset)

    def __getitem__(self, index):
        X = self.data[index][0]
        Y = self.data[index][1]
        return X, Y

    def __len__(self):
        return len(self.data)

