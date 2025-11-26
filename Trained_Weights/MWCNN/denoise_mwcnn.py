#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:52:47 2020

@author: dell
"""

import numpy as np
from torch import nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import time

from Packages.MWCNN.MWCNN_Nested_2group import MWCNN

def denoise_MWCNN(noisy, path, GPU=True):
    
  '''
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"              #选择显卡
  pynvml.nvmlInit()                                     #如需获取显卡显存占用状况则使用以下代码
  h = pynvml.nvmlDeviceGetHandleByIndex(0)              #获取显卡0的句柄
  
  while 1:
  	m = pynvml.nvmlDeviceGetMemoryInfo(h)
  	k=m.used                                          #得到显卡m的使用内存字节数
  	print(k)
  	if k < 1000000000:
  		time.sleep(10)
  		break
  	time.sleep(60)
  '''   
  noisy = noisy.astype(np.float32)
  #noisy.cuda()
  #print(noisy.dtype)  
  
  if GPU:
      if path=='/data/Wys/CS/LDAMP_python/ldamp/Packages/MWCNN/model/MWCNN_90_100.pth':
          net = MWCNN()
          net = torch.nn.DataParallel(net).cuda()
          net.load_state_dict(torch.load(path))
          noisy = torch.from_numpy(noisy)
          noisy = noisy.cuda() 
          x_hat = net(noisy)
          x_hat = x_hat.cpu().detach().numpy()
      else:
          net = MWCNN()
          net.load_state_dict(torch.load(path))
          net.cuda()  
          noisy = torch.from_numpy(noisy)
          noisy = noisy.cuda()        
          x_hat = net(noisy)
          x_hat = x_hat.cpu().detach().numpy()


  else:
    print('MWCNN model 12 must be putted on GPU!!!')
    '''
    net = MWCNN()
    net.load_state_dict(torch.load(path, map_location = torch.device('cpu')))
    noisy = Variable(torch.from_numpy(noisy))
    #noisy = noisy.float()
    #print(noisy.dtype)
    x_hat = net(noisy)
    x_hat = x_hat.detach().numpy().astype(np.float64)
    #print(x_hat)
    '''
  
  return x_hat