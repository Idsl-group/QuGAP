import math
import torch
import logging
import torch.nn as nn

from config import cfg
logger = logging.getLogger(__name__)

class Target_CNN(nn.Module):
    """Convolutional Neural Network"""
    def __init__(self, image_size = cfg.IMG_SIZE, in_channels = cfg.N_CHANNELS, num_classes = cfg.NUM_CLASSES):
        super(Target_CNN, self).__init__()
        self.im_size =  image_size
        self.im_channels = in_channels
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Conv2d(self.im_channels, 16, kernel_size = 3, padding = 1), nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),                                
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1), nn.ReLU(),               
            nn.MaxPool2d(kernel_size = 2, stride = 2),                                
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1), nn.ReLU(),               
            nn.MaxPool2d(kernel_size = 2, stride = 2),                                

            nn.Flatten(),
            nn.Linear(((self.im_size // 8)**2)*64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, self.num_classes)
        )
        logger.info(f'Creating Target Model: Classical Convolutional Neural Network (CONV)...')

    def forward(self, x):
        x = x.reshape(x.shape[0], self.im_channels, self.im_size, self.im_size)
        logits = self.model(x)
        probs = nn.Softmax(dim = 1)(logits)
        return logits, probs


class Gen_Uni(nn.Module):
    """Trainable Unitary Simulator"""
    def __init__(self, image_size = cfg.IMG_SIZE):
        super(Gen_Uni, self).__init__()
        self.im_size = image_size
        self.dim = 2**int(math.ceil(math.log2(image_size)))
        self.uni = nn.Parameter(torch.zeros(self.dim, self.dim, dtype = torch.cfloat))
        torch.nn.init.xavier_uniform_(self.uni)

    def forward(self, x):
        b_size = x.shape[0]
        u, _, v = torch.svd(self.uni)
        uni = u@(v.H)

        norm_x = torch.div(x, torch.norm(x, dim = 1).reshape(b_size, 1)).reshape(b_size, -1, 1)
        y = torch.matmul(uni.expand(b_size, self.dim, self.dim), norm_x).reshape(b_size, -1)
        norm_y = torch.div(y, torch.norm(y, dim = 1).reshape(b_size, 1)).reshape(b_size, -1)
        return norm_y


class Gen_FC(nn.Module):
    """Fully Connected Generator"""
    def __init__(self, z_dim = cfg.GEN_ZDIM, image_size = cfg.IMG_SIZE, out_channels = cfg.N_CHANNELS):
        super(Gen_FC, self).__init__()
        self.im_size = image_size
        self.im_channels = out_channels
        self.z_dim = z_dim

        self.models = nn.ModuleList([nn.Sequential(
            nn.Linear(self.z_dim, 256, bias=False), nn.BatchNorm1d(1), nn.LeakyReLU(0.2),
            nn.Linear(256, 512, bias=False), nn.BatchNorm1d(1), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024, bias=False), nn.BatchNorm1d(1), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(1), nn.LeakyReLU(0.2),
            nn.Linear(512, (self.im_size**2), bias=False),
            nn.Tanh(),
            nn.Unflatten(2, (self.im_size, self.im_size))
        ) for _ in range(self.im_channels)])
        
        logger.info(f'Creating generator 1 - Fully Connected...')

    def forward(self, x):
        out = [None for _ in range(self.im_channels)]
        for i in range(self.im_channels):
            out[i] = self.models[i](x)
        out = torch.squeeze(torch.stack(out, dim = 1), dim = 2)
        return out
    

class Gen_Conv(nn.Module):
    """Deep Convolutional Generator"""
    def __init__(self, z_dim = cfg.GEN_ZDIM, image_size = cfg.IMG_SIZE, out_channels = cfg.N_CHANNELS):
        super(Gen_Conv, self).__init__()
        self.z_dim = z_dim
        self.im_size = image_size
        self.im_channels = out_channels
        self.conv_blocks = int(math.ceil(math.log2((image_size + 3) // 4)))
        self.gen_channels = out_channels * (2**(self.conv_blocks - 1))

        self.models = nn.ModuleList([nn.Sequential() for _ in range(self.im_channels)])
        for c in range(self.im_channels):
            for i in range(self.conv_blocks + 1):
                if i == 0:
                    self.models[c].append(nn.ConvTranspose2d(self.z_dim, self.gen_channels, kernel_size = 4, stride = 1, padding = 0, bias=False))
                    self.models[c].append(nn.BatchNorm2d(self.gen_channels))
                    self.models[c].append(nn.LeakyReLU(0.2, inplace = True))
                elif i == self.conv_blocks:
                    self.models[c].append(nn.ConvTranspose2d(self.im_channels, self.im_channels, kernel_size = 4, stride = 2, padding = 1, bias=False))
                    self.models[c].append(nn.Tanh())
                else:
                    in_filters = int(self.gen_channels / (2**(i-1)))
                    out_filters = int(self.gen_channels / (2**i))
                    self.models[c].append(nn.ConvTranspose2d(in_filters, out_filters, kernel_size = 4, stride = 2, padding = 1, bias=False))
                    self.models[c].append(nn.BatchNorm2d(out_filters))
                    self.models[c].append(nn.LeakyReLU(0.2, inplace = True))
        logger.info(f'Creating generator 2 - Deep Convolutional...')

    def forward(self, x):
        out = [None for _ in range(self.im_channels)]
        x = torch.reshape(x, (x.shape[0], -1, 1, 1))
        for i in range(self.im_channels):
            out[i] = self.models[i](x)
        out = torch.squeeze(torch.stack(out, dim = 1), dim = 2)
        out = out[ : , : , : self.im_size, : self.im_size]
        return out