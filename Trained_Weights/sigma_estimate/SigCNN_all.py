import torch
from torch import nn
import scipy.io as sio

class SigCNN(nn.Module):
    def __init__(self):
        super(SigCNN, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        self.conva1 = nn.Conv2d(1, 64, 5, 1, 2, bias=False)
        self.conva2 = nn.Conv2d(64, 64, 5, 1, 2, bias=False)
        self.conva3 = nn.Conv2d(64, 64, 5, 1, 2, bias=False)
        self.conva4 = nn.Conv2d(64, 64, 5, 1, 2, bias=False)
        self.conva5 = nn.Conv2d(64, 64, 5, 1, 2, bias=False)
        
        #-------------------------------------
        self.conv1 = nn.Conv2d(64, 64, 5, 1, 2, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)

        self.conv7 = nn.Conv2d(256, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        s = 2
        a1 = self.conva1(x)
        a2 = self.relu(self.conva2(a1))
        a3 = self.conva3(a2)
        a3_1 = a3 + a1
        a4 = self.relu(self.conva4(a3_1))
        a5 = self.conva5(a4)
        a5_1 = a5 + a3_1
        a5_2 = a1 - a5_1

        c1 = self.relu(self.conv1(a5_2))
        c1 = c1[:, :, s-1::s, s-1::s]
        c2 = self.relu(self.conv2(c1))
        c2 = c2[:, :, s-1::s, s-1::s]
        c3 = self.relu(self.conv3(c2))
        c3 = c3[:, :, s-1::s, s-1::s]
        c4 = self.relu(self.conv4(c3))
        c4 = c4[:, :, s-1::s, s-1::s]
        c5 = self.relu(self.conv5(c4))
        c5 = c5[:, :, s-1::s, s-1::s]
        c6 = self.relu(self.conv6(c5))
        c6 = c6[:, :, s-1::s, s-1::s]
        c7 = torch.nn.functional.adaptive_avg_pool2d(c6, (1, 1)) #全局平均池化
        c8 = self.conv7(c7)
        output = torch.squeeze(c8,dim =2)
        output = torch.squeeze(output, dim=2)
        return output


if __name__ == '__main__':
    # 是否使用cuda
    path = 'D:\Chenzan\CS_image\Trained_Weights\sigma_estimate\sigcnn_from_tf_npz.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('#### Test Model ###')
    mat_data = sio.loadmat('A.mat')
    A_np = mat_data['noisy']
    A_torch = torch.from_numpy(A_np).float()
    x = A_torch.unsqueeze(0).unsqueeze(0)   
    x = x.to(device)   
    # x = torch.rand(4, 1, 256, 256).to(device)
    model = SigCNN().to(device)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    y = model(x)
    print(y)
