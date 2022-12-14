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
        return x

class TransConv_Block(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(num_features=out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return x

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
        return x

class MRN(nn.Module):
    def __init__(self, in_channels=3, class_num=3, multiple=1):
        super().__init__()
        self.in_channels = in_channels
        self.class_num   = class_num
        self.multiple    = multiple
        self.resize_list_origin = [8, 16, 32, 64, 128, 256]
        self.resize_list = [i//self.multiple for i in self.resize_list_origin]
        
        # 256
        self.block11_1 = Backbone_Block(in_channels, 16, 3, 1, 1)
        self.maxp11_2 = nn.MaxPool2d(2, 2, 0)
        # 128
        self.block21_1 = Backbone_Block(16, 32, 3, 1, 1)
        self.maxp21_2 = nn.MaxPool2d(2, 2, 0)
        # 64
        self.block31_1 = Backbone_Block(32, 64, 3, 1, 1)
        self.maxp31_2 = nn.MaxPool2d(2, 2, 0)
        # 32
        self.block41_1 = Backbone_Block(64, 128, 3, 1, 1)
        self.maxp41_2 = nn.MaxPool2d(2, 2, 0)
        
        self.block51_1 = Backbone_Block(128, 256, 3, 1, 1)
        self.ident52 = Identical_Block(512, 512, 1, 1, 0)

        self.trans42_1 = TransConv_Block(512, 256, 2, 2, 0)
        self.ident42_2 = Identical_Block(256, 256, 1, 1, 0)
        self.block42_3 = Backbone_Block(512, 256, 3, 1, 1)

        self.trans32_1 = TransConv_Block(256, 128, 2, 2, 0)
        self.ident32_2 = Identical_Block(128, 128, 1, 1, 0)
        self.block32_3 = Backbone_Block(256, 128, 3, 1, 1)

        self.trans22_1 = TransConv_Block(128, 64, 2, 2, 0)
        self.ident22_2 = Identical_Block(64, 64, 1, 1, 0)
        self.block22_3 = Backbone_Block(128, 64, 3, 1, 1)

        self.trans12_1 = TransConv_Block(64, 32, 2, 2, 0)
        self.ident12_2 = Identical_Block(32, 32, 1, 1, 0)
        self.block12_3 = Backbone_Block(64, 16, 3, 1, 1)

        self.ident00 = Identical_Block(16, 16, 1, 1, 0)

        self.output1 = nn.Conv2d(16, class_num, 1, 1, 0)
        self.output2 = nn.Softmax()

    def resolution_concat(self, tx, cx, devide, size):
        devide = devide + 1 if self.multiple == 2 else devide
        crop_resize_cx = F.interpolate(cx[:, :, cx.shape[2]//2-self.resize_list[devide]:cx.shape[2]//2+self.resize_list[devide], cx.shape[3]//2-self.resize_list[devide]:cx.shape[3]//2+self.resize_list[devide]], size=size)
        concat = torch.cat([tx, crop_resize_cx], dim=1)
        return concat

    def forward(self, x):
        tx, cx = x[:,0,...], x[:,1,...]
        
        # target encoder
        tx111 = self.block11_1(tx)
        tx112 = self.maxp11_2(tx111)
        tx211 = self.block21_1(tx112)
        tx212 = self.maxp21_2(tx211)
        tx311 = self.block31_1(tx212)
        tx312 = self.maxp31_2(tx311)
        tx411 = self.block41_1(tx312)
        tx412 = self.maxp41_2(tx411)
        tx51  = self.block51_1(tx412)

        # context encoder
        cx111 = self.block11_1(cx)
        cx112 = self.maxp11_2(cx111)
        cx211 = self.block21_1(cx112)
        cx212 = self.maxp21_2(cx211)
        cx311 = self.block31_1(cx212)
        cx312 = self.maxp31_2(cx311)
        cx411 = self.block41_1(cx312)
        cx412 = self.maxp41_2(cx411)
        cx51  = self.block51_1(cx412)

        # decoder
        x52_sc = self.resolution_concat(tx51, cx51, 0, 32)
        x52 = self.ident52(x52_sc)

        x42_1 = self.trans42_1(x52)
        x42_sc = self.resolution_concat(tx411, cx411, 1, 64)
        x42_21 = self.ident42_2(x42_sc)
        x42_22 = torch.cat([x42_1, x42_21], dim=1)
        x42_3 = self.block42_3(x42_22)

        x32_1 = self.trans32_1(x42_3)
        x32_sc = self.resolution_concat(tx311, cx311, 2, 128)
        x32_21 = self.ident32_2(x32_sc)
        x32_22 = torch.cat([x32_1, x32_21], dim=1)
        x32_3 = self.block32_3(x32_22)

        x22_1 = self.trans22_1(x32_3)
        x22_sc = self.resolution_concat(tx211, cx211, 3, 256)
        x22_21 = self.ident22_2(x22_sc)
        x22_22 = torch.cat([x22_1, x22_21], dim=1)
        x22_3 = self.block22_3(x22_22)

        x12_1 = self.trans12_1(x22_3)
        x12_sc = self.resolution_concat(tx111, cx111, 4, 512)
        x12_21 = self.ident12_2(x12_sc)
        x12_22 = torch.cat([x12_1, x12_21], dim=1)
        x12_3 = self.block12_3(x12_22)

        x00 = self.ident00(x12_3)

        output1 = self.output1(x00)
        output2 = self.output2(output1)

        return output2

def mrn(in_channels, class_num, multiple):
    return MRN(in_channels, class_num, multiple)