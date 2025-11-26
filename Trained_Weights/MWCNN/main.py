# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:31:31 2020

@author: Wys
"""

import argparse
import os
from math import log10
from torch import nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
import time
import scipy.io as sio
import scipy
from data_utils import *
from model.MWCNN_Nested_2group import MWCNN
import pynvml
import numpy as np

#仅含训练过程，测试仅需导入模型与参量即可


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"              #选择显卡

'''
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
img_size = 256
batch_size = 8               #设定batch大小
lr_init = 1e-4                #设定初始学习率
n_epoch_init = 300           #设定学习总epoch数
train_path = '/data/Wys/dataset/DIV2K_NEW/'
test_path = '/data/Wys/dataset//Set12/'

train_hr_img_list = sorted(load_file_list(path=train_path, regx='.*.png', printable=False))#得到相关文件夹下所有png文件名
test_img_list = sorted(load_file_list(path=test_path, regx='.*.png', printable=False))
n_test = len(test_img_list)
netG = MWCNN()                 #网络模型，测试时，采取netG = RCAN.eval()
mse_criterion = nn.MSELoss()   #loss设定
netG.load_state_dict(torch.load('MWCNN_80_90_1.pth'))#若有预训练模型则可导入
netG = torch.nn.DataParallel(netG).cuda()

mse_criterion.cuda()        #loss函数载入显卡

optimizerG = optim.Adam(netG.parameters(),lr=lr_init)#设定优化器需要更新的参数与学习率



for epoch in range(0,n_epoch_init+1):        #epoch训练大循环
    
    f = open('mseloss_90_100.txt','r+',encoding='UTF-8')
    f.read()
    
    netG.train()                             #训练设定
    random.shuffle(train_hr_img_list)        #打乱训练文件顺序
    epoch_time = time.time()                 #计时
    total_mse_loss1,total_mse_loss2,n_iter = 0,0,0
    if epoch > 60 and (epoch % 60 == 0):
        lr = lr_init/(3**((epoch-60)//60))
        log = " ** new learning rate: %f " % (lr_init/(3**((epoch-60)//60)))
        print(log)
        for param_group in optimizerG.param_groups:
            param_group['lr'] = lr                            #每过10个epoch更新一次学习率（epoch>20），学习率缩减三倍
    
    for idx in range(0, len(train_hr_img_list)//batch_size*batch_size, batch_size):      #epoch中batchsize小训练循环
        step_time = time.time()
        train_hr_imgs = threading_data(train_hr_img_list[idx:idx + batch_size], fn=get_imgs_fn,path=train_path)#读取图片数据
        train_batch_data= threading_data(train_hr_imgs,fn=crop_sub_imgs_fn,sl=90,sh=100,w=img_size,h=img_size,is_random=True)#图片数据截取与加噪
        [b_imgs_384,n_imgs_384]  = np.split(train_batch_data,2,axis=1)#噪声图与原图分离
        real_img = Variable(torch.from_numpy(b_imgs_384))  #将numpy转为tensor格式
        z = Variable(torch.from_numpy(n_imgs_384))
        if torch.cuda.is_available():
        	real_img = real_img.cuda()   #数据入显卡
        	z = z.cuda()
        netG.zero_grad()    #网络梯度清零，每次更新前需要清零操作
        fake_img = netG(z)  #反馈网络结果
        g_loss = mse_criterion(fake_img, real_img)  #计算loss
        g_loss.backward()  #反馈loss计算梯度
        optimizerG.step()   #更新参数
        print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, n_iter, time.time() - step_time, g_loss.item()))#打印loss
        total_mse_loss1 += g_loss.item()
        n_iter += 1
    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f\r\n" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss1 / n_iter)#打印loss
    print(log)
    f.write(log)
    #f.close()
       
    netG.eval()
    psnr = np.zeros(n_test)
    for idx in range(n_test):
        t_img, n_img = get_test_img(test_path + test_img_list[idx], 95)
        with torch.no_grad():
            
            denoise = netG(n_img)
            psnr[idx] = PSNR(t_img.cpu().numpy(), denoise.cpu().numpy())
            log = test_img_list[idx] + ": Epoch[%d] psnr = %.4f" % (epoch, psnr[idx])        
            print(log)        
            f.write(log)        
    mean_psnr = np.mean(psnr)
    print("mean_psnr = [%.4f]\r\n" %  mean_psnr)
    f.write("mean_psnr = [%.4f]\r\n" %  mean_psnr)
    f.close()
    torch.save(netG.state_dict(), 'MWCNN_90_100.pth')#保存模型
#f.close()
    
