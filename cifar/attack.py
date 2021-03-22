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
from dataset import prepare_dataset
from utils import str2bool, check_mkdir

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
    
    args = parser.parse_args()

    return args

def main(args):
    net = Net(args['num_classes'], args['conservative'], args['conservative_a'], args['triangular']).to(device)
    net.load_state_dict(torch.load(path + 'best_net_checkpoint.pt'))
# =============================================================================
#     trainset = prepare_dataset(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'train') 
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['train_batch_size'],
#                                               shuffle=True, num_workers=1)
# =============================================================================
    testset = prepare_dataset(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'test') 
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=1)

    for eps in np.arange(0,0.1,0.005):
        test_acc_attack= test(testloader, net, eps)
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

def test(test_loader, net, eps):
    net.eval()
    Acc_y = 0
    nb = 0
    class_correct = list(0. for i in range(hps['num_classes']))
    class_total = list(0. for i in range(hps['num_classes']))
    
    for i, data in enumerate(test_loader):
        X, Y = data 
        X = Variable(X).to(device)
        Y = Variable(Y.squeeze()).to(device) 
        
        loss = nn.CrossEntropyLoss()
        X = fgsm_attack(net, loss, X, Y, eps)
        nb = nb + len(X)

        outputs, _ = net(X)
        _, predicted = torch.max(outputs, 1)
        # print('test posterior:')
        # print(outputs.max(1)[0], outputs.max(1)[1])
        Acc_y = Acc_y + (predicted - Y).nonzero().size(0)
        c = (predicted == Y).squeeze()
        for i in range(len(X)):
            label = Y[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

  
    test_acc = (nb - Acc_y)/nb 
    print('Accuracy:', test_acc)
    
    for i in range(hps['num_classes']):
        print('at eps %.5f accuracy of %5s : %5f ' % (eps,
            classes[i], class_correct[i] / (1e-8 + class_total[i])))
        with open(path + 'attack_result_per_class.txt', 'a') as f:
            f.write('at eps %.5f accuracy of %5s : %5f \n' % (eps,
                classes[i], class_correct[i] / (1e-8 + class_total[i])))
    #print("test acc: %.5f"%test_acc)
    return test_acc




if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    for k in args.keys():
        hps[k] = args[k]
        
    if args['tune_hps']:
        path = 'tune_hps/conservative_a_' + str(args['conservative_a']) + \
                '/lr_' + str(args['lr']) + '/tbs_' + str(args['train_batch_size']) + '/wd_' + str(args['weight_decay']) + '/'
         
    if args['conservative'] == 'False':
        path = 'conservative_False/exp_' + str(args['exp']) + '/' 
    elif args['conservative'] == 'center':
        path = 'conservative_center/' + str(args['conservative_a']) + '/exp_' + str(args['exp']) + '/' 

    #check_mkdir(path)
    main(hps)

