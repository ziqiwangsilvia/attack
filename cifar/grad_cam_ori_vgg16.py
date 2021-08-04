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
import argparse
from network import Net
from utils import str2bool
from attack import fgsm_attack


hps = {'train_all': True,
       'train_index': [0,1],
       'test_all': True,
       'test_index': [0,1],
       'num_classes': 10,
       'train_batch_size': 128,
       'test_batch_size': 100,
       'epoch': 10,
       'lr': 1e-3,
       'print_freq':1,
       'conservative': 'False',
       'conservative_a': 0.1,
       'exp': 0}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conservative', default='False', choices = ['False', 'center'])
    parser.add_argument('--conservative_a', default= 0.1, type=float)
    parser.add_argument('--exp', default=0, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=5e-6, type=float)
    parser.add_argument('--tune_hps', default=False, type=str2bool)
    parser.add_argument('--triangular', default=False, type=str2bool)
    parser.add_argument('--network', default='vgg16', choices=['vgg16', 'vgg19'])
    
    args = parser.parse_args()

    return args


# =============================================================================
# # use the ImageNet transformation
# transform = transforms.Compose([transforms.Resize((224, 224)), 
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# 
# # define a 1 image dataset
# dataset = datasets.ImageFolder(root='data/test_gradcam/', transform=transform)
# # define the dataloader to load that single image
# dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
# =============================================================================

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

# define the dataloader to load that single image
dataloader = torch.utils.data.DataLoader(dataset, batch_size=30,
                                         shuffle=False, num_workers=1)


class VGG(nn.Module):
    def __init__(self, args):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        #self.vgg = torchvision.models.vgg16(pretrained=True).to(device)
        self.vgg = Net(args['network'], args['num_classes'], args['conservative'], args['conservative_a'], args['triangular']).to(device)
# =============================================================================
#         # get the pretrained VGG19 network
#         self.vgg = torchvision.models.vgg16(pretrained=True).to(device)
#         # disect the network to access its last convolutional layer
#         self.features_conv = self.vgg.features[:36]
#         
#         # get the max pool of the features stem
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
#         # get the classifier of the vgg19
#         self.classifier = self.vgg.classifier
#         
#         # placeholder for the gradients
#         self.gradients = None
# =============================================================================
        self.vgg.load_state_dict(torch.load(path + 'best_net_checkpoint.pt'))
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.net.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # get the classifier of the vgg19
        self.classifier = self.vgg.net.classifier
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
        z = self.classifier(x)
        return z, z
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
    
def get_heatmap(vgg, img, label, args):       
    # set the evaluation mode
    vgg.eval()
    
    # get the image from the dataloader

    # get the most likely prediction of the model
    outputs, _ = vgg(img)
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
    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu()
    
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    return heatmap


def matplotlib_imshow(img, i, args, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().numpy()
    if one_channel:
        plt.axis('off')
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(args['path'] + 'test_%d_eps_%.2f.jpg'%(i, args['eps']), bbox_inches='tight')
        
def save_exp(heatmap, img, args, i):
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    img = np.uint8(255 * img.squeeze().resize(32, 32, 3).detach().cpu().numpy())
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 1 + img*0
    final = cv2.resize(superimposed_img, (128, 128))
    cv2.imwrite(args['path'] + 'map_%d_eps_%.2f.jpg'%(i, args['eps']), final)

def grad_cam(args):
    vgg = VGG(args)
    loss = nn.CrossEntropyLoss()
    vgg.eval()
    for eps in np.arange(0,0.35,0.05):
        args['eps'] = eps
        images, labels = next(iter(dataloader))
        for i, (img, label) in enumerate(zip(images, labels)):
            img = img.unsqueeze(0).to(device)    
            label = label.unsqueeze(0).to(device) 
            img_attack= fgsm_attack(vgg, loss, img, label, eps)        
           
            heatmap = get_heatmap(vgg, img_attack.to(device), label, args)
            # create grid of images
            img_grid = torchvision.utils.make_grid(img)
            # show images
            matplotlib_imshow(img_grid, i, args, one_channel=False)
            save_exp(heatmap, img, args, i)

if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    for k in args.keys():
        hps[k] = args[k]
        
    if args['tune_hps']:
        path = 'tune_hps/conservative_a_' + str(args['conservative_a']) + \
                '/lr_' + str(args['lr']) + '/tbs_' + str(args['train_batch_size']) + '/wd_' + str(args['weight_decay']) + '/'
         
    elif args['conservative'] == 'False':
        path = 'conservative_False/exp_' + str(args['exp']) + '/' 
    elif args['conservative'] == 'center':
        path = 'conservative_center/' + str(args['conservative_a']) + '/exp_' + str(args['exp']) + '/' 
        
# =============================================================================
#     path = 'data/test_gradcam/Elephant/'
# =============================================================================
    hps['path'] = path
    hps['eps'] = 0
    grad_cam(hps)
