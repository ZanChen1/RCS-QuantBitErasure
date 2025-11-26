import math

import torch
import torch.nn as nn
import torch.nn.functional as F

### Convolution Block Attention Module

class Flatten(nn.Module):  #像素拉平1维，为了输入MLP学习
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):   #通道注意力
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels   #特征通道数
        self.mlp = nn.Sequential(    #MLP
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_pool = F.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_avg = self.mlp(avg_pool)
        max_pool = F.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_max = self.mlp(max_pool)

        channel_att_sum = channel_avg + channel_max

        scale = torch.sigmoid(channel_att_sum).unsqueeze(    # unsqueeze 给tensor增加一个维度，增加维度然后扩展成和原图像相同大小相乘
            2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):  #空间注意力中使用的通道Maxpool和avgpool
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = torch.nn.Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))  #单路卷积，通道2变1，不改变大小

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16,no_spatial=False):  # pool_types=['avg', 'max']   no_spatial是否取消空间attention
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio)  # pool_types
        self.SpatialGate = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

