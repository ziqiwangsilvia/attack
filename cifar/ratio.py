#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:23:25 2021

@author: ziqi
"""
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter










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
from dataset import prepare_dataset, prepare_dataset_cifar100
from utils import str2bool, check_mkdir

hps = {'train_all': True,
       'train_index': [0,1],
       'test_all': True,
       'test_index': [0,1],
       'num_classes': 100,
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
    parser.add_argument('--conservative', default='False', choices = ['False', 'center', 'double', 'marco'])
    parser.add_argument('--conservative_a', default= 0.2, type=float)
    parser.add_argument('--exp', default=0, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=5e-6, type=float)
    parser.add_argument('--tune_hps', default=False, type=str2bool)
    parser.add_argument('--triangular', default=False, type=str2bool)
    parser.add_argument('--attack_type', default='FGSM', choices = ['FGSM', 'BIM'])
    parser.add_argument('--network', default='vgg16', choices=['vgg16', 'vgg19', 'resnet18', 'resnet50'])
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    
    args = parser.parse_args()

    return args

def main(args):
    if args['dataset'] == 'cifar10':
        args['num_classes'] = 10
    else:
        args['num_classes'] = 100
    net = Net(args['network'], args['num_classes'], args['conservative'], args['conservative_a'], args['triangular']).to(device)
    net.load_state_dict(torch.load(path + 'best_net_checkpoint.pt'))

    if args['dataset'] == 'cifar10':
        testset = prepare_dataset(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'test') 
    else:
        testset = prepare_dataset_cifar100(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'test') 
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=1)
    ratio_mm = ratio(testloader, net)
    plot_ratio(ratio_mm)



def ratio(loader, net):
    ratio = []
    net.eval()
    for data in loader:
        images, labels = data
        images.requires_grad = True

        outputs, z = net(images)

        # get the margin between the top 2 classes
        values, indexes = torch.topk(z, 2, dim=1, largest=True)
        margin_sub = values[:,0] - values[:,1]
        
        # get the gradient magnitude
        attacks = []
        for i in range(len(z)):
            net.zero_grad()
            z[i][indexes[i][0]].backward(retain_graph=True)
            g_max = images.grad[i]

            net.zero_grad()
            images.grad=None
            z[i][indexes[i][1]].backward(retain_graph=True)
            g_min = images.grad[i]
            
#             attack = torch.sum(torch.abs(g_min - g_max), (1,2))
            attack = torch.norm(g_max-g_min).unsqueeze(0)
            attacks.append(attack)
        attacks = torch.cat(attacks)        
  
        ratio_image = attacks/margin_sub  
        ratio.extend(ratio_image.detach().view(-1).numpy())
        
    return ratio

def plot_ratio(ratio):
    fig, ax = plt.subplots()
    plt.hist(ratio, bins=100, range = (0,1))
    plt.xlabel('ratio', fontsize=20)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(fontsize=20)
    plt.ylabel('Ours', fontsize=20)
    plt.yticks(fontsize=20)
    # plt.savefig('marco_l2_ratio_37.pdf', bbox_inches='tight')
    
    
    
if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    for k in args.keys():
        hps[k] = args[k]
        
    if args['tune_hps']:
        path = 'tune_hps_NLLLoss_' + args['conservative'] + '_' + args['network'] + '_' + args['dataset'] + '/conservative_a_' + str(args['conservative_a']) + \
                '/lr_' + str(args['lr']) + '/tbs_' + str(args['train_batch_size']) + '/wd_' + str(args['weight_decay']) + '/'
    elif args['conservative'] == 'double':
        path = 'conservative_double/exp_' + str(args['exp']) + '/'   
    elif args['conservative'] == 'marco':
        path = 'conservative_marco/exp_' + str(args['exp']) + '/' 
    elif args['conservative'] == 'False':
        path = 'conservative_False/exp_' + str(args['exp']) + '/' 
    elif args['conservative'] == 'center':
        path = 'conservative_center/' + str(args['conservative_a']) + '/exp_' + str(args['exp']) + '/' 
    print(path)
    hps['path'] = path
    main(hps)














    