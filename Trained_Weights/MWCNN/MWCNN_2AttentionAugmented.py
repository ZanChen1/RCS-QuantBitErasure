import sys
sys.path.append("..")
import common
import CBAM
import torch
import torch.nn as nn
import scipy.io as sio
import torch.nn.functional as F


class AugmentedConv(nn.Module):  #relative暂时为false 暂时未进行相对编码
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):  #K=2V=0.25  Nh=4或8
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = attn_out.transpose(2, 3)  # B Nh dv/Nh H*W     源代码没有，直接reshape会将HW改变，在此加上
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class AugmentedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1,act=nn.ReLU(True)):  #K=2V=0.2  Nh=4或8
        super(AugmentedBlock, self).__init__()
        m = []
        m.append(AugmentedConv(in_channels, out_channels, kernel_size, dk, dv, Nh, shape=shape, relative=relative, stride=stride))
        m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x



# Example Code
# tmp = torch.randn((16, 3, 32, 32)).to(device)
# augmented_conv1 = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, padding=1, stride=2, shape=16).to(device)
# conv_out1 = augmented_conv1(tmp)
# print(conv_out1.shape)
#
# for name, param in augmented_conv1.named_parameters():
#     print('parameter name: ', name)
#
# augmented_conv2 = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, padding=1, stride=1, shape=32).to(device)
# conv_out2 = augmented_conv2(tmp)
# print(conv_out2.shape)



class RDB_Conv(nn.Module):  #RDB内部卷积
    def __init__(self, inChannels, growRate, kSize=3):   #growth rate，每一个 block 里面的卷积层送到下一个卷积层的特征的数量
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):    #RDB块G0为初始输入特征通道数，G为每个卷积块为下一个卷积块提供的特征通道数;由于dense连接，后面的卷积块的输入会逐渐增加G
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)  #过1×1卷积进行局部特征融合

    def forward(self, x):
        return self.LFF(self.convs(x)) + x  #RDB局部残差连接


def make_model(parent=False):
    return MWCNN()

class MWCNNgroup(nn.Module):
    def __init__(self,conv,n_feats,kernel_size,act = nn.ReLU(True)):
        super(MWCNNgroup,self).__init__()

        self.DWT = common.DWT()
        self.IWT = common.IWT()


        d_l0 = [common.BBlock(conv, n_feats, n_feats, kernel_size, act=act)]
        #d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))
        d_l0 = []
        d_l0.append(RDB(n_feats, n_feats // 2, 2)) # 64 32 2

        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]  #common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)
        #d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)) #d = 2 , 1
        d_l1.append(RDB(n_feats * 2, n_feats, 2))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))#common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False)
        #d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(RDB(n_feats * 4, n_feats * 2, 2))

        pro_l3 = []
        pro_l3.append(AugmentedBlock(n_feats * 16,n_feats * 8,kernel_size,dk=n_feats * 2 ,dv=n_feats ,Nh=8)) #common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False)
        # pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))#d = 2 , 3
        pro_l3.append(RDB(n_feats * 8, n_feats * 4, 2))

        # pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))#d = 3 , 2
        pro_l3.append(RDB(n_feats * 8, n_feats * 4, 2))
        pro_l3.append(AugmentedBlock(n_feats * 8,n_feats * 16,kernel_size,dk=n_feats * 4 ,dv=n_feats * 2,Nh=8)) #common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False)


        # i_l2 = [common.DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)] #d = 2 , 1
        i_l2 = [RDB(n_feats * 4, n_feats * 2, 2)]
        i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False)) #common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False)

        # i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1 = [RDB(n_feats * 2, n_feats , 2)]
        i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False)) #common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False)

        # i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]
        i_l0 = [RDB(n_feats , n_feats // 2 , 2)]

        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)

    def forward(self, x):
        x0 = self.d_l0(x)  # 三层之后
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x = self.i_l0(x_) + x  # group最外层残差

        return x


class MWCNN(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(MWCNN, self).__init__()
        n_resblocks = 20 #args.n_resblocks
        n_feats = 64 #args.n_feats
        kernel_size = 3  # 3
        self.scale_idx = 0
        nColor = 1 #args.n_colors
        act = nn.ReLU(True)

        MWCNN_head = [conv(nColor,n_feats,kernel_size)]

        MWCNN_group1 = [MWCNNgroup(conv,n_feats,kernel_size)]
        MWCNN_group2 = [MWCNNgroup(conv,n_feats,kernel_size)]

        MWCNN_fuse = [conv(2 * n_feats, n_feats, kernel_size=1)]

        MWCNN_CBAM = [CBAM.CBAM(n_feats)]
        MWCNN_tail = [conv(n_feats,nColor,kernel_size)]

        self.head = nn.Sequential(*MWCNN_head)
        self.group1 = nn.Sequential(*MWCNN_group1)
        self.group2 = nn.Sequential(*MWCNN_group2)
        self.CBAM = nn.Sequential(*MWCNN_CBAM)
        self.fuse = nn.Sequential(*MWCNN_fuse)
        self.tail = nn.Sequential(*MWCNN_tail)

    def forward(self, x):

        x_head = self.head(x)
        x_res1 = self.group1(x_head)
        x_res2 = self.group2(x_res1)
        x_res = torch.cat((x_res1,x_res2),1) #128
        x_res = self.fuse(x_res) #64

        x_res = self.CBAM(x_res)
        x_res = x_res + x_head
        x_res = self.tail(x_res)

        x_clean = x + x_res #最外层残差

        return x_clean

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
