"""
MRN(Multi-Resolution Network)
description: The same model as the structure in the paper. (https://arxiv.org/pdf/1807.09607.pdf)
Number of input image: 2 images with the same size and center but with different resolution (Target: high resolution, Context: low resolution)
size: (2, 3, 512, 512) -> (512, 512)
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch.utils import model_zoo

### MRN

class Backbone_Block(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(num_features=out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return(x)

class Identical_Block(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(num_features=out_c)
        self.ident = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.ident(x)
        return(x)

class MRN(nn.Module):
    def __init__(self, in_channels=3, class_num=3, multiple=1):
        super().__init__()
        self.multiple = multiple
        self.resize_list_origin = [8, 16, 32, 64, 128]
        self.resize_list = [i//self.multiple for i in self.resize_list_origin]
        self.block11_1 = Backbone_Block(in_channels, 16, 3, 1, 1)
        self.maxp11_2 = nn.MaxPool2d(2, 2, 0)

        self.block21_1 = Backbone_Block(16, 32, 3, 1, 1)
        self.maxp21_2 = nn.MaxPool2d(2, 2, 0)

        self.block31_1 = Backbone_Block(32, 64, 3, 1, 1)
        self.maxp31_2 = nn.MaxPool2d(2, 2, 0)

        self.block41_1 = Backbone_Block(64, 128, 3, 1, 1)
        self.maxp41_2 = nn.MaxPool2d(2, 2, 0)

        self.block51 = Backbone_Block(128, 256, 3, 1, 1)
        
        self.ident52 = Identical_Block(256, 256, 1, 1, 0)

        self.trans42_1 = nn.ConvTranspose2d(512, 256, 2, 2, 0)
        self.ident42_2 = Identical_Block(256, 256, 1, 1, 0)
        self.block42_3 = Backbone_Block(512, 128, 3, 1, 1)

        self.trans32_1 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.ident32_2 = Identical_Block(128, 128, 1, 1, 0)
        self.block32_3 = Backbone_Block(256, 64, 3, 1, 1)

        self.trans22_1 = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.ident22_2 = Identical_Block(64, 64, 1, 1, 0)
        self.block22_3 = Backbone_Block(128, 32, 3, 1, 1)

        self.trans12_1 = nn.ConvTranspose2d(32, 32, 2, 2, 0)
        self.ident12_2 = Identical_Block(32, 32, 1, 1, 0)
        self.block12_3 = Backbone_Block(64, 16, 3, 1, 1)

        self.output1 = nn.Conv2d(16, class_num, 1, 1, 0)
        self.output2 = nn.Softmax()

    def encoder(self, x, context=None, skip=None):
        x = x[:,1,...] if context else x[:,0,...]
        x11_1 = self.block11_1(x)
        if skip == 1:
            return x11_1
        x11_2 = self.maxp11_2(x11_1)
        
        x21_1 = self.block21_1(x11_2)
        if skip == 2:
            return x21_1
        x21_2 = self.maxp21_2(x21_1)
        
        x31_1 = self.block31_1(x21_2)
        if skip == 3:
            return x31_1
        x31_2 = self.maxp31_2(x31_1)

        x41_1 = self.block41_1(x31_2)
        if skip == 4:
            return x41_1
        x41_2 = self.maxp41_2(x41_1)

        x51 = self.block51(x41_2)
        return x51

    def forward(self, x):
        x51 = self.encoder(x)
        x52 = self.ident52(x51)
        y51 = self.encoder(x, context=True)
        y52 = F.interpolate(y51[:, :, y51.shape[2]//2-self.resize_list[0]:y51.shape[2]//2+self.resize_list[0], y51.shape[3]//2-self.resize_list[0]:y51.shape[3]//2+self.resize_list[0]], size=32)
        x5 = torch.cat([x52, y52], dim=1)

        x42_1 = self.trans42_1(x5)
        x42_2 = self.ident42_2(x42_1)
        y41_1 = self.encoder(x, skip=4)
        y42_1 = self.encoder(x, context=True, skip=4)
        y42_2 = F.interpolate(y42_1[:, :, y42_1.shape[2]//2-self.resize_list[1]:y42_1.shape[2]//2+self.resize_list[1], y42_1.shape[3]//2-self.resize_list[1]:y42_1.shape[3]//2+self.resize_list[1]], size=64)
        x4 = torch.cat([x42_2, y41_1, y42_2], dim=1)
        x42_3 = self.block42_3(x4)

        x32_1 = self.trans32_1(x42_3)
        x32_2 = self.ident32_2(x32_1)
        y31_1 = self.encoder(x, skip=3)
        y32_1 = self.encoder(x, context=True, skip=3)
        y32_2 = F.interpolate(y32_1[:, :, y32_1.shape[2]//2-self.resize_list[2]:y32_1.shape[2]//2+self.resize_list[2], y32_1.shape[3]//2-self.resize_list[2]:y32_1.shape[3]//2+self.resize_list[2]], size=128)
        x3 = torch.cat([x32_2, y31_1, y32_2], dim=1)
        x32_3 = self.block32_3(x3)
        
        x22_1 = self.trans22_1(x32_3)
        x22_2 = self.ident22_2(x22_1)
        y21_1 = self.encoder(x, skip=2)
        y22_1 = self.encoder(x, context=True, skip=2)
        y22_2 = F.interpolate(y22_1[:, :, y22_1.shape[2]//2-self.resize_list[3]:y22_1.shape[2]//2+self.resize_list[3], y22_1.shape[3]//2-self.resize_list[3]:y22_1.shape[3]//2+self.resize_list[3]], size=256)
        x2 = torch.cat([x22_2, y21_1, y22_2], dim=1)
        x22_3 = self.block22_3(x2)

        x12_1 = self.trans12_1(x22_3)
        x12_2 = self.ident12_2(x12_1)
        y11_1 = self.encoder(x, skip=1)
        y12_1 = self.encoder(x, context=True, skip=1)
        y12_2 = F.interpolate(y12_1[:, :, y12_1.shape[2]//2-self.resize_list[4]:y12_1.shape[2]//2+self.resize_list[4], y12_1.shape[3]//2-self.resize_list[4]:y12_1.shape[3]//2+self.resize_list[4]], size=512)
        x1 = torch.cat([x12_2, y11_1, y12_2], dim=1)
        x12_3 = self.block12_3(x1)

        output1 = self.output1(x12_3)
        output2 = self.output2(output1)
        return output2

def mrn(in_channels, class_num, multiple):
    return MRN(in_channels, class_num, multiple)