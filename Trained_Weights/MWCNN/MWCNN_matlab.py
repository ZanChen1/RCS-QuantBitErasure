#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:00:30 2020

@author: 
"""

import torch
from MWCNN_Nested_2group import MWCNN
#from MWCNN_2AttentionAugmented import MWCNN
import cv2
import numpy as np
import array


def denoise_MWCNN_17(noisy, path): 
    net = MWCNN()
    net.load_state_dict(torch.load(path))
    net.cuda()  
    net.eval()
    for k, v in net.named_parameters():
        v.requires_grad = False  
    x_hat = net(noisy)  
    return x_hat


def denoise17(noisy, highth, width, sigma_hat):    
    
    mwcnn_path = '..\\Trained_Weights\\MWCNN\\'
    if sigma_hat>500:
        path = mwcnn_path + 'MWCNN_500_1000.pth'
    elif sigma_hat>300:
        path = mwcnn_path + 'MWCNN_300_500.pth'
    elif sigma_hat>150:
        path = mwcnn_path + 'MWCNN_150_300.pth'
    else:
        path = mwcnn_path + 'MWCNN_150_300.pth'
    
    noisy = np.array(noisy)
    noisy = torch.from_numpy(noisy)
    noisy = noisy.cuda()
    noisy = noisy/255
    noisy = torch.reshape(noisy, (1, 1, int(highth), int(width)))
    noisy = noisy.float()
    x_hat = denoise_MWCNN_17(noisy, path)
    x_hat = x_hat.double()        
    x_hat = x_hat*255
    x_hat = torch.reshape(x_hat, (int(highth) * int(width), 1))
 
    x_hat = x_hat.cpu().numpy()
    x_hat = array.array('d',x_hat)
    return x_hat


