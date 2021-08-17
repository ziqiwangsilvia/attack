#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:02:50 2021

@author: ziqi
"""

import torch
import torch.nn as nn 
from torch import optim
from torch.autograd import Variable

import argparse
from utils import AverageMeter, str2bool, check_mkdir
from network import Net
from dataset import prepare_dataset, prepare_dataset_cifar100

hps = {'train_all': True,
       'train_index': [0,1],
       'test_all': True,
       'test_index': [0,1],
       'num_classes': 100,
       'train_batch_size': 128,
       'test_batch_size': 100,
       'epoch': 300,
       'lr': 1e-3,
       'weight_decay': 5e-6,
       'print_freq':1,
       'conservative': False,
       'conservative_a': 0.1,
       'exp': 0,
       'triangular': False}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conservative', default='False', choices = ['False', 'center', 'double', 'marco'])
    parser.add_argument('--conservative_a', default= 0.2, type=float)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--exp', default=0, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=5e-6, type=float)
    parser.add_argument('--tune_hps', default=False, type=str2bool)
    parser.add_argument('--triangular', default=False, type=str2bool)
    parser.add_argument('--network', default='vgg16', choices=['vgg16', 'vgg19', 'resnet18', 'resnet50', 'squeezenet'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    
    args = parser.parse_args()

    return args

def main(args):
    net = Net(args['network'], args['num_classes'], args['conservative'], args['conservative_a'], args['triangular']).to(device)
    if args['dataset'] == 'cifar10':
        trainset = prepare_dataset(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'train') 
        testset = prepare_dataset(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'test') 
    else:
        trainset = prepare_dataset_cifar100(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'train') 
        testset = prepare_dataset_cifar100(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'test') 
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['train_batch_size'],
                                              shuffle=True, num_workers=1)   
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=1)

    criterion = nn.CrossEntropyLoss()
    #momentum=0.9, 
    optimizer = optim.Adam(net.parameters(), lr=args['lr'],
                      weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_Acc = 0 

    for epoch in range(1, args['epoch'] + 1):
        train_acc = train(trainloader, net, criterion, optimizer, epoch, args)
        test_acc= test(testloader, net)   
        scheduler.step()
        with open(path + 'cifar_train_accuracy.txt', 'a')as f:
            f.write('[epoch %d], train_accuracy is: %.5f\n' % (epoch, train_acc))
        with open(path + 'cifar_test_accuracy.txt', 'a')as f:
            f.write('[epoch %d], test_accuracy is: %.5f\n' % (epoch, test_acc))
        if best_Acc < test_acc:
            best_Acc = test_acc 
            torch.save(net.state_dict(), path + 'best_net_checkpoint.pt')
    return best_Acc


def train(train_loader, net, criterion, optimizer, epoch, args):
    net.train()
    train_loss = AverageMeter()
    Acc_v = 0
    nb = 0

    for i, data in enumerate(train_loader):
        X, Y = data 
        X = Variable(X).to(device)
        Y = Variable(Y).squeeze().to(device)
        N = len(X)
        nb = nb+N

        outputs, _ = net(X)
        # print('max train posterior:')
        # print(outputs.max(1)[0])
        Acc_v = Acc_v + (outputs.argmax(1) - Y).nonzero().size(0)
        loss = criterion(outputs, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data.item(), N)     
        if epoch % args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (epoch, i + 1, len(train_loader), train_loss.avg))

    train_acc = (nb - Acc_v)/nb
    # print("train acc: %.5f"%train_acc)
    return train_acc

def test(test_loader, net):
    net.eval()
    Acc_y = 0
    nb = 0
    for i, data in enumerate(test_loader):
        X, Y = data 
        X = Variable(X).to(device)
        Y = Variable(Y.squeeze()).to(device) 
        nb = nb + len(X)

        outputs, _ = net(X)
        # print('test posterior:')
        # print(outputs.max(1)[0], outputs.max(1)[1])
        Acc_y = Acc_y + (outputs.argmax(1) - Y).nonzero().size(0)
  
    test_acc = (nb - Acc_y)/nb 
    #print("test acc: %.5f"%test_acc)
    return test_acc





if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    for k in args.keys():
        hps[k] = args[k]
         
    if args['tune_hps']:
        path = 'tune_hps_' + args['conservative'] + '_' + args['network'] + '_' + args['dataset'] + '/conservative_a_' + str(args['conservative_a']) + \
                '/lr_' + str(args['lr']) + '/tbs_' + str(args['train_batch_size']) + '/wd_' + str(args['weight_decay']) + '/'
    
    elif args['conservative'] == 'double':
        path = 'conservative_double/exp_' + str(args['exp']) + '/' 
    elif args['conservative'] == 'marco':
        path = 'conservative_marco/exp_' + str(args['exp']) + '/' 
    elif args['conservative'] == 'False':
        path = 'conservative_False/exp_' + str(args['exp']) + '/' 
    elif args['conservative'] == 'center':
        path = 'conservative_center/' + str(args['conservative_a']) + '/exp_' + str(args['exp']) + '/' 
    check_mkdir(path)
    best_acc= main(hps)
    
