#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:45:53 2021

@author: ziqi
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse

from torch.autograd import Variable
from network import Net, marco_softmax
from torchvision.models.resnet import resnet18, resnet50
from dataset import prepare_dataset, prepare_dataset_cifar100, Imagenette
from imagenet_tfrecord import ImageNet_TFRecord
from utils import str2bool, check_mkdir

import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from imagenet_finetune import accuracy, reduce_tensor, to_python_float, AverageMeter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conservative', default='marco', choices = ['False', 'center', 'double', 'marco'])
    parser.add_argument('--conservative_a', default= 0.2, type=float)
    parser.add_argument('--exp', default=0, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=5e-6, type=float)
    parser.add_argument('--tune_hps', default=False, type=str2bool)
    parser.add_argument('--triangular', default=False, type=str2bool)
    parser.add_argument('--attack_type', default='FGSM', choices = ['FGSM', 'BIM'])
    parser.add_argument('--network', default='resnet50', choices=['vgg16', 'vgg19', 'resnet18', 'resnet50'])
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'imagenette'])
    parser.add_argument('data', metavar='DIR', nargs='*', help='path(s) to dataset',
                        default='/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/imagenet/tfrecords')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers per GPU (default: 2)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    args = parser.parse_args()

    return args

def main(args):
    net = resnet50(pretrained=False)
    net = nn.Sequential(net, marco_softmax(1000))
    checkpoint = torch.load(args.path  + 'checkpoint.pth.tar')
    checkpoint['state_dict'] = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
    net.load_state_dict(checkpoint['state_dict'])

# =============================================================================
#     if args.dataset == 'imagenette':
#         args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         net.to(args.device)
#         if args.conservative == 'False':
#             criterion = nn.CrossEntropyLoss()
#         elif args.conservative == 'marco':
#             criterion = nn.NLLLoss()
#         valset = Imagenette(mode='val', input_shape=args.input_shape)
#         valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size,
#                                          shuffle=False, num_workers=1)
#         for eps in np.arange(0,1.1,0.1):
#             test_acc_attack= test_singel_proc(valloader, criterion, net, eps, args)
#             with open(args.path + 'imagenette_attack_result_all.txt', 'a') as f:
#                 f.write('acc at eps %.5f: %.5f \n' %(eps, test_acc_attack))
# =============================================================================
    if args.dataset == 'imagenet':
        # set address for master process to localhost since we use a single node
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
        # use all gpus pytorch can find
        args.world_size = torch.cuda.device_count()
        print('Found {} GPUs:'.format(args.world_size))
        for i in range(args.world_size):
            print('{} : {}'.format(i, torch.cuda.get_device_name(i)))
    
        # total batch size = batch size per gpu * ngpus
        args.total_batch_size = args.world_size * args.batch_size
    
        # TODO: find out what this stuff does
        print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
        cudnn.benchmark = True
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
        if not len(args.data):
            raise Exception("error: No data set provided")
      
        args.model = net
        # start processes for all gpus
        mp.spawn(gpu_process, nprocs=args.world_size, args=(args,))
        

def gpu_process(gpu, args):
        # each gpu runs in a separate proces
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                             rank=gpu, world_size=args.world_size)
    
        # Set cudnn to deterministic setting
        if args.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.manual_seed(gpu)
            torch.set_printoptions(precision=10)
    
        # push model to gpu
        model = args.model.cuda(gpu)
    
        # Scale learning rate based on global batch size
        args.lr = args.lr*float(args.test_batch_size*args.world_size)/256.
    
        # Use DistributedDataParallel for distributed training
        model = DDP(model, device_ids=[gpu], output_device=gpu)
    
        # define loss function (criterion) and optimizer
        criterion = nn.NLLLoss().cuda(gpu)
  
        # Data loading code
        valloader = ImageNet_TFRecord(args.data, 'val', args.test_batch_size, args.workers,
                                       gpu, args.world_size, augment=False)
    
        # only evaluate model, no training
        for eps in np.arange(0,1.1,0.1):
            test_acc_t1, test_acc_t5= test_multi_proc(valloader, model, criterion, gpu, args, eps)
            with open(args.path + 'tfimagenet_attack_result_all.txt', 'a') as f:
                f.write('acc at eps %.5f: %.5f, %.5f\n' %(eps, test_acc_t1, test_acc_t5))    
            
        return

def test_multi_proc(val_loader, model, criterion, gpu, args, eps):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda(gpu).long()
        val_loader_len = int(val_loader._size / args.test_batch_size)

        # compute output
        with torch.no_grad():
            if args.attack_type == 'FGSM':
                X = fgsm_attack(model, criterion, input, target, eps)
            elif args.attack_type == 'BIM':
                X = BIM_attack(model, criterion, input, target, 0, eps, 1, iters=100)
            output = model(X)
            loss = criterion(torch.log(output), target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss.data, args.world_size)
        prec1 = reduce_tensor(prec1, args.world_size)
        prec5 = reduce_tensor(prec5, args.world_size)
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if gpu == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, val_loader_len,
                       args.world_size * args.test_batch_size / batch_time.val,
                       args.world_size * args.test_batch_size / batch_time.avg,
                       batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def fgsm_attack(model, loss, images, labels, eps) :
    
    images = images
    labels = labels
    images.requires_grad = True
            
    outputs = model(images)

    model.zero_grad()
    if args.conservative == 'False':
        cost = loss(outputs, labels)
    elif args.conservative == 'marco':
        cost = loss(torch.log(outputs), labels)
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
        outputs= model(images)

        model.zero_grad()
        if args.conservative == 'False':
            cost = loss(outputs, labels)
        elif args.conservative == 'marco':
            cost = loss(torch.log(outputs), labels)
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


def test_singel_proc(test_loader, loss, net, eps, args):
    net.eval()
    Acc_y = 0
    nb = 0
    
    for i, data in enumerate(test_loader):
        X, Y = data 
        X = Variable(X).to(args.device)
        Y = Variable(Y.squeeze()).to(args.device) 
        
        if args.attack_type == 'FGSM':
            X = fgsm_attack(net, loss, X, Y, eps)
        elif args.attack_type == 'BIM':
            X = BIM_attack(net, loss, X, Y, 0, eps, 1, iters=100)
        nb = nb + len(X)

        outputs = net(X)
        _, predicted = torch.max(outputs, 1)
        # print('test posterior:')
        # print(outputs.max(1)[0], outputs.max(1)[1])
        Acc_y = Acc_y + (predicted - Y).nonzero().size(0)

  
    test_acc = (nb - Acc_y)/nb 
    print('Accuracy:', test_acc)

    return test_acc




if __name__ == '__main__':
    args = get_args()
    args.path = 'imagenet_exps/finetune_' + str(args.lr) + '/'  
    args.input_shape=(224, 224)
    print(args.path)
    #check_mkdir(path)
    main(args)

