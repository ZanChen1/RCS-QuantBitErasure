

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from SigCNN_all import SigCNN
import array


def sigma(noisy, img_weight, img_height):
    
    
    
    path = '..\\Trained_Weights\\sigma_estimate\\SigCNN.pth'
    noisy = np.array(noisy)
    noisy = torch.from_numpy(noisy)
    noisy = noisy.cuda()
    noisy = torch.reshape(noisy, (1, 1, int(img_weight), int(img_height)))
    noisy = noisy.float()
    net = SigCNN().cuda()
    net.load_state_dict(torch.load(path))
    net.eval()
    x_hat = net(noisy)
    
    x_hat = x_hat.double()
    x_hat = x_hat.cpu()
    x_hat = x_hat.detach().numpy()
    x_hat = array.array('d', x_hat)
    return x_hat









