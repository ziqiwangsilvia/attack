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
from utils import AverageMeter, str2bool
from network import Net
from dataset import prepare_dataset

hps = {'train_all': True,
       'train_index': [0,1],
       'test_all': True,
       'test_index': [0,1],
       'num_classes': 10,
       'train_batch_size': 32,
       'test_batch_size': 16,
       'epoch': 10,
       'lr': 1e-3,
       'print_freq':1,
       'conservative': False,
       'conservative_a': 0.1}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conservative', default=False, choices = ['monotone', 'center'])
    parser.add_argument('--conservative_a', default= 0.1, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    args = parser.parse_args()

    return args

def main(args):
    net = Net(args['num_classes'], args['conservative'], args['conservative_a']).to(device)
    trainset = prepare_dataset(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'train') 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['train_batch_size'],
                                              shuffle=True, num_workers=1)
    testset = prepare_dataset(args['train_all'], args['train_index'], args['test_all'], args['test_index'], 'test') 
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=1)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])

    best_Acc = 0 
    best_net = net
    for epoch in range(1, args['epoch'] + 1):
        train_acc = train(trainloader, net, criterion, optimizer, epoch, args)
        test_acc= test(testloader, net)     
        with open('cifar_train_accuracy.txt', 'a')as f:
            f.write('[epoch %d], train_accuracy is: %.5f\n' % (epoch, train_acc))
        with open('cifar_test_accuracy.txt', 'a')as f:
            f.write('[epoch %d], test_accuracy is: %.5f\n' % (epoch, test_acc))
        if best_Acc < test_acc:
            best_Acc = test_acc 
            best_net = net
    return best_Acc, best_net, testloader, trainloader


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
         
    best_acc, net, testloader, trainloader = main(hps)
    torch.save(net.state_dict(), 'conservative' + str(hps['conservative']) + '_checkpoint.pt')