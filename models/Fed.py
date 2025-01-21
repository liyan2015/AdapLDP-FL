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
    tensor1 = tensor1.numpy()
    tensor2 = tensor2.numpy()
    # product of two tensors
    elementwise_product = tensor1 * tensor2
    # Compute the norm of the tensors
    norm_tensor1 = np.linalg.norm(tensor1)
    norm_tensor2 = np.linalg.norm(tensor2)
    # Angle between the two tensors (in degrees)
    angle_rad = np.arccos(elementwise_product / (norm_tensor1 * norm_tensor2 + 0.0001))
    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)
    # -1 if the angle is obtuse, 1 if the angle is acute
    result = np.ones_like(angle_deg)  # Create a new array to store the updated values
    alpha = 0.02
    for index, value in np.ndenumerate(angle_deg):
        result[index] = torch.tensor(alpha * np.cos(value)) 
    return torch.tensor(result)

def update_dict_values(my_dict):
    count_minus1 = 0
    count_1 = 0
    for key in my_dict.keys():
        if torch.eq(my_dict[key], torch.tensor(-1.0)).all():
            count_minus1 += 1
        elif torch.eq(my_dict[key], torch.tensor(1.0)).all():
            count_1 += 1
    if count_minus1 > count_1:
        for key in my_dict.keys():
            if torch.eq(my_dict[key], torch.tensor(1)).all():
                my_dict[key] = torch.tensor(-1)
    else:
        for key in my_dict.keys():
            if torch.eq(my_dict[key], torch.tensor(-1)).all():
                my_dict[key] = torch.tensor(1)
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
    tilde_wi = ModelSub(copy.deepcopy(wi_t),copy.deepcopy(wi_t_1))  
    tilde_wg =  ModelSub(copy.deepcopy(wg_t_1),copy.deepcopy(wg_t_2))
    w_res = copy.deepcopy(tilde_wi)
    for k in wi_t.keys():
        w_res[k] = Ang_matrix(tilde_wi[k], tilde_wg[k])
    return copy.deepcopy(w_res)


def rotation_model(w, I_i):
    w_res = copy.deepcopy(w)
    w_rtt = copy.deepcopy(w)
    for k in w_res.keys():
        w_res[k] = torch.mul(w[k], I_i[k])
        w_rtt[k] = torch.add(w_rtt[k], w_res[k])
    return w_rtt
