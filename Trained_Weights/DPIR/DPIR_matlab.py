import os.path
import sys
import logging
import numpy as np
from collections import OrderedDict
import torch
from models.network_unet import UNetRes as net
from DPIR_utils import utils_logger 
from DPIR_utils import utils_model 
from DPIR_utils import utils_image as util
import array


def denoiser(img_L, img_weight, img_height, noise_level_model):
    x8 = False  
#---------------------------------- model injection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'drunet_gray'  # drunet_gray, PC_DRUNet_gray or SPC_DRUNet_gray
    if 'color' in model_name: 
            n_channels = 3                   # 3 for color image
    else:
        n_channels = 1                   # 1 for grayscale image
    model_pool = '..\\Trained_Weights\\DPIR\\'             # fixed
    model_path = os.path.join(model_pool, model_name+'.pth')
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
# -------------------------------------- data preprocess
    img_L = np.array(img_L)
    img_L = np.reshape(img_L, (int(img_weight), int(img_height)))
    if img_L.ndim == 2:
            img_L = np.expand_dims(img_L, axis=2)
    img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
    img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
    img_L = img_L.to(device)
    
    if not x8 and img_L.size(2)//8==0 and img_L.size(3)//8==0:
        img_E = model(img_L)
    elif not x8 and (img_L.size(2)//8!=0 or img_L.size(3)//8!=0):
        img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
    elif x8:
        img_E = utils_model.test_mode(model, img_L, mode=3)

    img_E = img_E.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img_E.ndim == 3:
        img_E = np.transpose(img_E, (1, 2, 0))
    img_E = img_E*255.0
    img_E = np.reshape(img_E, (int(img_weight)*int(img_height), 1))
    img_E = array.array('d', img_E)
    return img_E

    
    
