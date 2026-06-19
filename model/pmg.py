import torch
import torch.nn as nn
import torch.nn.functional as F
class discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size):
        super(discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        # self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel, self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'): 

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj


class Spa_Spe_Randomization(nn.Module):
    def __init__(self, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)  

    def forward(self, x, ):
        N, C, L, H, W = x.size()
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
        #doubt
            idx_swap = torch.randperm(N)  
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]  
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, L, H, W)

        return x, idx_swap


class Generator_3DCNN_SupCompress_pca(nn.Module):

    def __init__(self, imdim=3, imsize=[13, 13], device=0, dim1=128, dim2=8):
        super().__init__()

        self.patch_size = imsize[0]

        self.n_channel = dim2
        self.n_pca = dim1

        # 2D_CONV
        self.conv_pca = nn.Conv2d(imdim, self.n_pca, 1, 1) 

        self.inchannel = self.n_pca

        # 3D_CONV
        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=self.n_channel,
                               kernel_size=(3, 3, 3))

        
        self.Spa_Spe_Random = Spa_Spe_Randomization(device=device)

        # 
        self.conv6 = nn.ConvTranspose3d(in_channels=self.n_channel, out_channels=1, kernel_size=(3, 3, 3))

        # 2D_CONV
        self.conv_inverse_pca = nn.Conv2d(self.n_pca, imdim, 1, 1)

    def forward(self, x):
        x = self.conv_pca(x)

        x = x.reshape(-1, self.patch_size, self.patch_size, self.inchannel, 1)  # (256,48,13,13,1)
        x = x.permute(0, 4, 3, 1, 2)  # (256,1,48,13,13)

        x = F.relu(self.conv1(x))

        x, idx_swap = self.Spa_Spe_Random(x)

        x = torch.sigmoid(self.conv6(x))

        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(-1, self.inchannel, self.patch_size, self.patch_size)

        x = self.conv_inverse_pca(x)
        return x
    
def downsample(img, m):
    b, total_channels, height, width = img.shape
    
    
    group_channels = total_channels // m
    remainder = total_channels % m
    
  
    reduced_img = torch.zeros(b, m, height, width, device=img.device, dtype=img.dtype)
    end_channel = -1

    for i in range(m):
        start_channel = end_channel+1
        end_channel = start_channel + group_channels-1 + (1 if remainder > 0 else 0)
        remainder -= 1
        reduced_img[:, i, :, :] = img[:, start_channel:end_channel, :, :].mean(dim=1)
    return reduced_img
#doubt 
class Generator(nn.Module):
    def __init__(self, imdim=48, patch_size=13, layers = [], dim1 = 128, dim2 = 8, device=0):
        super().__init__()
        self.patch_size = patch_size
        self.n_channel = imdim
        self.layers_num = len(layers)
        self.dims = layers
        self.conv_pcas = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.sub_g = nn.ModuleList()
        for i in range(self.layers_num-1):
            self.conv_pcas.append(nn.Conv2d(in_channels=self.n_channel, out_channels=self.dims[i], kernel_size=1))
        for i in range(self.layers_num-1):
            self.upsamples.append(nn.Conv2d(in_channels=self.dims[i], out_channels=self.dims[i+1], kernel_size=1) )
        for i in range(self.layers_num):
            self.sub_g.append(Generator_3DCNN_SupCompress_pca(imdim=self.dims[i], imsize=[13, 13], device=device, dim1=dim1, dim2=dim2))
        
    def forward(self, x, current_step = 9999):
        if current_step <= self.layers_num:
            if current_step > 1:
                self.sub_g[current_step-2].requires_grad_(False)
                self.upsamples[current_step-2].requires_grad_(False)
                self.conv_pcas[current_step-2].requires_grad_(False)
            if current_step < self.layers_num:
                x_down = self.conv_pcas[current_step-1](x)
            else:
                x_down = x
            if len(self.dims) > 1:
                x_g = self.sub_g[0](downsample(x, self.dims[0]))#starting with downsampled input for the first generator
            else:
                x_g = self.sub_g[0](x)
            for i in range(1, current_step):
                x_g = self.sub_g[i](self.upsamples[i-1](x_g))#ending with upsampled output for the last generator
            return x_g, x_down
        else:
            if current_step < self.layers_num*2:
                x_down = self.conv_pcas[current_step-1-self.layers_num](x)
            else:
                x_down = x
            return x_down
            
class Dis(nn.Module):
    def __init__(self, imdim=48, patch_size=13, layers = [], proj=128, num_classes=7):
        super().__init__()
        self.patch_size = patch_size
        self.n_channel = imdim
        self.layers_num = len(layers)
        self.dims = layers
        # self.dims = [int((imdim+layers_num)/layers_num)*(i+1) for i in range(layers_num-1)]+[imdim]
        self.sub_d = nn.ModuleList()
        for i in range(self.layers_num):
            self.sub_d.append(discriminator(inchannel=self.dims[i], outchannel=proj, num_classes=num_classes, patch_size=self.patch_size))
        
    def forward(self, x, current_step = 9999, mode='train'):
        
        if current_step <= self.layers_num:
            if current_step > 1:
                self.sub_d[current_step-2].requires_grad_(False)
            clss, proj = self.sub_d[current_step-1](x, mode=mode)
            return clss, proj
        else:
            clss = self.sub_d[-1](x, mode='test')
            return clss
        
        