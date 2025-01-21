#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Author: YGF
# Des: AdapLDP-FL.

from dataclasses import replace
from cv2 import DRAW_MATCHES_FLAGS_DEFAULT
import matplotlib
from regex import W
from soupsieve import select
from sqlalchemy import false
from sympy import rad
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import random
from collections import deque
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import time
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg, NetSub, Dir_mat, rotation_model
from models.test import test_img


if __name__ == '__main__':
    time_start = time.time()
    # parse args
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)

    else:
        exit('Error: unrecognized model')
    # This is some set about DP parameters.
    dp_epsilon = args.dp_epsilon / (args.frac * args.epochs)
    dp_delta = args.dp_delta
    dp_mechanism = args.dp_mechanism
    dp_clip = args.dp_clip

    net_glob.train()  # server initialization
    w_glob = net_glob.state_dict()  # obtained model para to share
    net_temp = copy.deepcopy(net_glob)  # obtained model para to share
    w_g_0 = copy.deepcopy(w_glob) 

    all_clients = list(range(args.num_users))  # give each client a label (max = 99)!

    acc_test = []  # storge accuracy
    loss_training = []  # storge loss
    learning_rate = [args.lr for i in range(args.num_users)] # 100 client's learning rate are 0.01 
    
    print('Random number of clients is {}'.format(str(max(int(args.frac * args.num_users), 1))))

    scaler_c_i = [0.001 for i in range(args.num_users)] # scaler for every client. 
    scaler_c_g = 0.001  # 100 clients (1/100)
    local_K = 1  #local epochs
    wi_t_1 = [copy.deepcopy(w_glob) for i in range(args.num_users)] # preserve last model for all clients. 
    wi_t = [copy.deepcopy(w_glob) for i in range(args.num_users)]

    glo_queue = deque(maxlen=2)
    glo_queue.extend([w_glob, w_glob]) # Storing Global Models


    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m_tal = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m_tal, replace=False)
        begin_index = iter % (1 / args.frac)
        idxs_clients = all_clients[int(begin_index * args.num_users * args.frac):
                                   int((begin_index + 1) * args.num_users * args.frac)]
        
        for idx in idxs_users:
            args.lr = learning_rate[idx]
            wi_t_1[idx] = copy.deepcopy(wi_t[idx])  # obtain model of last round

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                                dp_epsilon=dp_epsilon, dp_delta=dp_delta,
                                dp_mechanism=dp_mechanism, dp_clip=dp_clip)
            w, loss, curLR = local.train(net=copy.deepcopy(net_glob).to(args.device))  # download global parameters
            learning_rate[idx] = curLR
            
            loss_locals.append(copy.deepcopy(loss))

            #computer local scaler
            if iter > 0:
                scaler_c_i[idx] = 0.6*(abs(scaler_c_i[idx]-scaler_c_g)) + 0.4*1/(local_K*learning_rate[idx])\
                              *NetSub(copy.deepcopy(w_glob), copy.deepcopy(w))
            # First define a new model for storing the local model parameters  
            net_temp.load_state_dict(w)  # important step
            
            w = local.add_noise(net=copy.deepcopy(net_temp), \
                                scaler_c_i = scaler_c_i[idx])   
            wi_t[idx] = copy.deepcopy(w)  # Update the model of the previous round


            if iter > 0:
                 Ii_t = Dir_mat(wi_t=copy.deepcopy(w), wi_t_1=copy.deepcopy(wi_t_1[idx]), \
                             wg_t_1=copy.deepcopy(glo_queue[1]), wg_t_2=copy.deepcopy(glo_queue[0]))
                 w = rotation_model(copy.deepcopy(w), copy.deepcopy(Ii_t))
        
            w_locals.append(copy.deepcopy(w))   # return each client's trained para
        # server side: model agg and parameter 'c_i' meaning   
        w_glob = FedAvg(w_locals)   # FedAvg
        glo_queue.popleft()    # delete element
        glo_queue.append(copy.deepcopy(w_glob))   # add element
        
        # normalize 
        scaler_c_i = [(x - np.mean(scaler_c_i)) / (np.std(scaler_c_i)+0.00001) for x in scaler_c_i]
        scaler_c_g = sum(scaler_c_i) / len(scaler_c_i)   # Mean for ci
    
        net_glob.load_state_dict(w_glob)   # copy weight to net_glob
        # print accuracy while training
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        print("Round {:3d},Testing accuracy: {:.2f}".format(iter+1, acc_t))
        acc_test.append(acc_t.item())
        
        # Training loss but not testing loss
        loss_t = sum(loss_locals) / len(loss_locals) # if delete this sentence, it's testing loss.
        loss_training.append(loss_t)

    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(rootpath + '/AdapDP_accfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon), "w")
    lossfile = open(rootpath + '/AdapDP_lossfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()
    for ls in loss_training:
        loss = str(ls)
        lossfile.write(loss)
        lossfile.write('\n')
    lossfile.close()

    time_stop = time.time()
    time_sum = time_stop-time_start
    print('运行时间: {} 分钟!'.format(time_sum/60))
