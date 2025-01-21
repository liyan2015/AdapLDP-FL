#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from pyparsing import originalTextFor
from sympy import comp
import torch
import numpy as np
import torch


def normalize(ls): 
    s = sum(ls) 
    return [i/s for i in ls] 

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# norm and subtraction for Network model 
def NetSub(w1, w2):  
    w_sub = copy.deepcopy(w1)
    for k in w_sub.keys():
        w_sub[k] = torch.subtract(w1[k], w2[k])
    # Merge all weight matrices into a NumPy array.
    weights_array = np.concatenate([v.flatten() for v in w_sub.values()])
    # Compute the biparadigm of the weight array
    norm = np.linalg.norm(weights_array, ord=2)
    return norm
    
# subtraction for Network model 
def ModelSub(w1, w2):  
    w_sub = copy.deepcopy(w1)
    for k in w_sub.keys():
        w_sub[k] = torch.subtract(w1[k], w2[k])
    return w_sub


def Ang_matrix(tensor1, tensor2):
    return torch.tensor(result)

def update_dict_values(my_dict):

    return my_dict

def update_tensor_values(my_tensor):
    count_minus1 = 0
    count_1 = 0
    for value in my_tensor:
        if torch.eq(value, torch.tensor(-1.0)).all():
            count_minus1 += 1
        elif torch.eq(value, torch.tensor(1.0)).all():
            count_1 += 1
    if count_minus1 > count_1:
        for i in range(len(my_tensor)):
            if torch.eq(my_tensor[i], torch.tensor(1)).all():
                my_tensor[i] = torch.tensor(-1)
    else:
        for i in range(len(my_tensor)):
            if torch.eq(my_tensor[i], torch.tensor(-1)).all():
                my_tensor[i] = torch.tensor(1)
    return my_tensor


# direction matrix 
def Dir_mat(wi_t, wi_t_1, wg_t_1, wg_t_2):
    return copy.deepcopy(w_res)


def rotation_model(w, I_i):
    w_res = copy.deepcopy(w)
    w_rtt = copy.deepcopy(w)
    for k in w_res.keys():
        w_res[k] = torch.mul(w[k], I_i[k])
        w_rtt[k] = torch.add(w_rtt[k], w_res[k])
    return w_rtt
