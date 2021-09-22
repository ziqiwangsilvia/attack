#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:45:53 2021

@author: ziqi
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
from network import Net
from torchvision.models.resnet import resnet18, resnet50
from dataset import prepare_dataset, prepare_dataset_cifar100, Imagenette
from utils import str2bool, check_mkdir

hps = {'train_all': True,
       'train_index': [0,1],
       'test_all': True,
       'test_index': [0,1],
       'num_classes': 1000,
       'input_shape':(224, 224),
       'train_batch_size': 128,
       'test_batch_size': 100,
       'epoch': 10,
       'lr': 1e-3,
       'print_freq':1,
       'conservative': 'False',
       'conservative_a': 0.1,
       'exp': 0}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conservative', default='marco', choices = ['False', 'center', 'double', 'marco'])
    parser.add_argument('--conservative_a', default= 0.2, type=float)
    parser.add_argument('--exp', default=0, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=5e-6, type=float)
    parser.add_argument('--tune_hps', default=False, type=str2bool)
    parser.add_argument('--triangular', default=False, type=str2bool)
    parser.add_argument('--attack_type', default='FGSM', choices = ['FGSM', 'BIM'])
    parser.add_argument('--network', default='resnet50', choices=['vgg16', 'vgg19', 'resnet18', 'resnet50'])
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    
    args = parser.parse_args()

    return args

def main(args):
    net = Net(args['network'], args['num_classes'], args['conservative'], args['conservative_a'], args['triangular']).to(device)
    checkpoint = torch.load(path  + 'checkpoint.pth.tar')
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(epoch)

    valset = Imagenette(mode='val', input_shape=hps['input_shape'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=1)

    for eps in np.arange(0,1.1,0.1):
        test_acc_attack= test(valloader, net, eps, args)
        with open(path + 'attack_result_all.txt', 'a') as f:
            f.write('acc at eps %.5f: %.5f \n' %(eps, test_acc_attack))


def fgsm_attack(model, loss, images, labels, eps) :
    
    images = images
    labels = labels
    images.requires_grad = True
            
    outputs, _ = model(images)

    model.zero_grad()
    cost = loss(outputs, labels)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()    
    return attack_images

def BIM_attack(model, loss, images, labels, scale, eps, alpha, iters=0) :
    images = images
    labels = labels
    clamp_max = 255
    
    if iters == 0 :
        # The paper said min(eps + 4, 1.25*eps) is used as iterations
        iters = int(min(eps + 4, 1.25*eps))
                
    if scale :
        eps = eps / 255
        clamp_max = clamp_max / 255
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs,r = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        attack_images = images + (eps/iters)*images.grad.sign()
        
        # Clip attack images(X')
        # min{255, X+eps, max{0, X-eps, X'}}
        # = min{255, min{X+eps, max{max{0, X-eps}, X'}}}
        
        # a = max{0, X-eps}
        a = torch.clamp(attack_images - eps, min=0)
        # b = max{a, X'}
        b = (attack_images>=a).float()*attack_images + (a>attack_images).float()*a
        # c = min{X+eps, b}
        c = (b > attack_images+eps).float()*(attack_images+eps) + (attack_images+eps >= b).float()*b
        # d = min{255, c}
        images = torch.clamp(c, max=clamp_max).detach_()
            
    return images

def test(test_loader, net, eps, args):
    net.eval()
    Acc_y = 0
    nb = 0
# =============================================================================
#     class_correct = list(0. for i in range(args['num_classes']))
#     class_total = list(0. for i in range(args['num_classes']))
# =============================================================================
    
    for i, data in enumerate(test_loader):
        X, Y = data 
        X = Variable(X).to(device)
        Y = Variable(Y.squeeze()).to(device) 
        
        if args['conservative'] == 'False':
            loss = nn.CrossEntropyLoss()
        elif args['conservative'] == 'marco':
            loss = nn.NLLLoss()
        if args['attack_type'] == 'FGSM':
            X = fgsm_attack(net, loss, X, Y, eps)
        elif args['attack_type'] == 'BIM':
            X = BIM_attack(net, loss, X, Y, 0, eps, 1, iters=100)
        nb = nb + len(X)

        outputs, _ = net(X)
        _, predicted = torch.max(outputs, 1)
        # print('test posterior:')
        # print(outputs.max(1)[0], outputs.max(1)[1])
        Acc_y = Acc_y + (predicted - Y).nonzero().size(0)
# =============================================================================
#         c = (predicted == Y).squeeze()
#         for i in range(len(X)):
#             label = Y[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
# =============================================================================

  
    test_acc = (nb - Acc_y)/nb 
    print('Accuracy:', test_acc)
    
# =============================================================================
#     for i in range(hps['num_classes']):
#         print('at eps %.5f accuracy of %5s : %5f ' % (eps,
#             classes[i], class_correct[i] / (1e-8 + class_total[i])))
#         with open(path + 'attack_result_per_class.txt', 'a') as f:
#             f.write('at eps %.5f accuracy of %5s : %5f \n' % (eps,
#                 classes[i], class_correct[i] / (1e-8 + class_total[i])))
#     #print("test acc: %.5f"%test_acc)
# =============================================================================
    return test_acc




if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    for k in args.keys():
        hps[k] = args[k]
        
    path = 'imagenet_exps/finetune_' + str(args['lr']) + '/'
    
    print(path)
    #check_mkdir(path)
    hps['path'] = path
    main(hps)

