#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:58:47 2021

@author: ziqi
"""
import cv2
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# use the ImageNet transformation
transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

# define a 1 image dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

# define the dataloader to load that single image
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                         shuffle=False, num_workers=1)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = torchvision.models.vgg16(pretrained=True).to(device)
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:30]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
def get_heatmap(img, label):    
# initialize the VGG model
    vgg = VGG()
    
    # set the evaluation mode
    vgg.eval()
    
    # get the image from the dataloader

    # get the most likely prediction of the model
    outputs = vgg(img)
    pos, pred = outputs.max(dim=1)
    print(label, pred)
    # get the gradient of the output with respect to the parameters of the model
    pos.backward()
    
    # pull the gradients out of the model
    gradients = vgg.get_activations_gradient()
    
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # get the activations of the last convolutional layer
    activations = vgg.get_activations(img).detach()
    
    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    return heatmap


def matplotlib_imshow(img, i, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.axis('off')
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig('./test_%d.jpg'%i, bbox_inches='tight')
        
def save_exp(heatmap, img, i):
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    img = np.uint8(255 * img.squeeze().resize(32, 32, 3).numpy())
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 1 + img*0
    final = cv2.resize(superimposed_img, (128, 128))
    cv2.imwrite('./map_%d.jpg'%i, final)

def main():
    images, labels = next(iter(dataloader))
    images.to(device)
    labels.to(device)
    for i, (img, label) in enumerate(zip(images, labels)):
        img = img.unsqueeze(0)
        heatmap = get_heatmap(img, label)
        # create grid of images
        img_grid = torchvision.utils.make_grid(img)
        # show images
        matplotlib_imshow(img_grid, i, one_channel=False)
        save_exp(heatmap, img, i)

if __name__ == '__main__':
    main()