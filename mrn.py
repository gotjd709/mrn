"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch.utils import model_zoo

### MRN backbone seresnext101

pretrained_settings = {
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 512, 512],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 3
        }
    },
}

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

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, class_num=3, multiple=1):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.multiple = multiple
        self.resize_list_origin = [4, 8, 16, 32, 64]
        self.resize_list = [i//self.multiple for i in self.resize_list_origin]
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        self.layer00 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer01 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(
            block,
            planes=16,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=32,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=64,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=128,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.ident52 = Identical_Block(512, 256, 1, 1, 0)

        self.trans42_1 = nn.ConvTranspose2d(768, 256, 2, 2, 0)
        self.ident42_2 = Identical_Block(256, 256, 1, 1, 0)
        self.block42_3 = Backbone_Block(768, 128, 3, 1, 1)

        self.trans32_1 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.ident32_2 = Identical_Block(128, 128, 1, 1, 0)
        self.block32_3 = Backbone_Block(384, 64, 3, 1, 1)

        self.trans22_1 = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.ident22_2 = Identical_Block(64, 64, 1, 1, 0)
        self.block22_3 = Backbone_Block(192, 32, 3, 1, 1)

        self.trans12_1 = nn.ConvTranspose2d(32, 32, 2, 2, 0)
        self.ident12_2 = Identical_Block(32, 32, 1, 1, 0)
        self.block12_3 = Backbone_Block(160, 16, 3, 1, 1)
        
        self.decode_layer0_1 = nn.ConvTranspose2d(16, 16, 2, 2, 0)
        self.decode_layer0_2 = Backbone_Block(16, 16, 3, 1, 1)
        self.decode_layer0_3 = Backbone_Block(16, 16, 3, 1, 1)

        self.output1 = nn.Conv2d(16, class_num, 1, 1, 0)
        self.output2 = nn.Softmax()
        
    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def encoder(self, x, context=None, skip=None):
        x11_1 = self.layer00(x[:,1,...]) if context else self.layer00(x[:,0,...])
        if skip == 1:
            return x11_1
        x11_2 = self.layer01(x11_1)
        
        x21 = self.layer1(x11_2)
        if skip == 2:
            return x21
        
        x31 = self.layer2(x21)
        if skip == 3:
            return x31

        x41 = self.layer3(x31)
        if skip == 4:
            return x41

        x51 = self.layer4(x41)
        return x51

    def forward(self, x):
        x51 = self.encoder(x)
        x52 = self.ident52(x51)
        y51 = self.encoder(x, context=True)
        y52 = F.interpolate(y51[:, :, y51.shape[2]//2-self.resize_list[0]:y51.shape[2]//2+self.resize_list[0], y51.shape[3]//2-self.resize_list[0]:y51.shape[3]//2+self.resize_list[0]], size=16)
        x5 = torch.cat([x52, y52], dim=1)

        x42_1 = self.trans42_1(x5)
        x42_2 = self.ident42_2(x42_1)
        y41_1 = self.encoder(x, skip=4)
        y42_1 = self.encoder(x, context=True, skip=4)
        y42_2 = F.interpolate(y42_1[:, :, y42_1.shape[2]//2-self.resize_list[1]:y42_1.shape[2]//2+self.resize_list[1], y42_1.shape[3]//2-self.resize_list[1]:y42_1.shape[3]//2+self.resize_list[1]], size=32)
        x4 = torch.cat([x42_2, y41_1, y42_2], dim=1)
        x42_3 = self.block42_3(x4)

        x32_1 = self.trans32_1(x42_3)
        x32_2 = self.ident32_2(x32_1)
        y31_1 = self.encoder(x, skip=3)
        y32_1 = self.encoder(x, context=True, skip=3)
        y32_2 = F.interpolate(y32_1[:, :, y32_1.shape[2]//2-self.resize_list[2]:y32_1.shape[2]//2+self.resize_list[2], y32_1.shape[3]//2-self.resize_list[2]:y32_1.shape[3]//2+self.resize_list[2]], size=64)
        x3 = torch.cat([x32_2, y31_1, y32_2], dim=1)
        x32_3 = self.block32_3(x3)
        
        x22_1 = self.trans22_1(x32_3)
        x22_2 = self.ident22_2(x22_1)
        y21_1 = self.encoder(x, skip=2)
        y22_1 = self.encoder(x, context=True, skip=2)
        y22_2 = F.interpolate(y22_1[:, :, y22_1.shape[2]//2-self.resize_list[3]:y22_1.shape[2]//2+self.resize_list[3], y22_1.shape[3]//2-self.resize_list[3]:y22_1.shape[3]//2+self.resize_list[3]], size=128)
        x2 = torch.cat([x22_2, y21_1, y22_2], dim=1)
        x22_3 = self.block22_3(x2)

        x12_1 = self.trans12_1(x22_3)
        x12_2 = self.ident12_2(x12_1)
        y11_1 = self.encoder(x, skip=1)
        y12_1 = self.encoder(x, context=True, skip=1)
        y12_2 = F.interpolate(y12_1[:, :, y12_1.shape[2]//2-self.resize_list[4]:y12_1.shape[2]//2+self.resize_list[4], y12_1.shape[3]//2-self.resize_list[4]:y12_1.shape[3]//2+self.resize_list[4]], size=256)
        x1 = torch.cat([x12_2, y11_1, y12_2], dim=1)
        x12_3 = self.block12_3(x1)
        
        x01 = self.decode_layer0_1(x12_3)
        x02 = self.decode_layer0_2(x01)
        x03 = self.decode_layer0_3(x02)

        output1 = self.output1(x03)
        output2 = self.output2(output1)
        return output2

def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

def mrn_seresnext101(class_num, multiple, pretrained=None):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  class_num=class_num, multiple=multiple)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

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