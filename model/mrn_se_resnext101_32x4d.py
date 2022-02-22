"""
MRN(Multi-Resolution Network)
description: The same model as the structure in the paper. (https://arxiv.org/pdf/1807.09607.pdf)
Number of input image: 2 images with the same size and center but with different resolution (Target: high resolution, Context: low resolution)
size: (2, 3, 284, 284) -> (70, 70)

ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo

__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


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
                 downsample_padding=1, num_classes=1000):
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
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


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

def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


"""

Block models

"""

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


"""

MRN (Multi-Resolution Network) + Backbone: SE-ResNeXt101_32x4d

"""


class MRN(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.layer00 = se_resnext101_32x4d().__dict__['_modules']['layer0'][:-1]
        self.layer01 = se_resnext101_32x4d().__dict__['_modules']['layer0'][-1]
        self.layer10 = se_resnext101_32x4d().__dict__['_modules']['layer1'][0]
        self.layer11 = se_resnext101_32x4d().__dict__['_modules']['layer1'][1:]
        self.layer20 = se_resnext101_32x4d().__dict__['_modules']['layer2'][0]
        self.layer21 = se_resnext101_32x4d().__dict__['_modules']['layer2'][1:]
        self.layer30 = se_resnext101_32x4d().__dict__['_modules']['layer3'][0]
        self.layer31 = se_resnext101_32x4d().__dict__['_modules']['layer3'][1:]
        self.layer40 = se_resnext101_32x4d().__dict__['_modules']['layer4'][0]
        self.layer41 = se_resnext101_32x4d().__dict__['_modules']['layer4'][1:]

        self.trans52 = nn.ConvTranspose2d(2048, 256, 2, 2, 0)
        self.ident53 = Identical_Block(2304, 256, 1, 1, 0)

        self.trans42_1 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.trans42_2 = nn.ConvTranspose2d(1024, 128, 2, 2, 0)
        self.ident42_3 = Identical_Block(1152, 256, 1, 1, 0)
        self.block42_4 = Backbone_Block(384, 128, 3, 1, 1)

        self.trans32_1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.trans32_2 = nn.ConvTranspose2d(512, 64, 2, 2, 0)
        self.ident32_3 = Identical_Block(576, 128, 1, 1, 0)
        self.block32_4 = Backbone_Block(192, 64, 3, 1, 1)

        self.trans22_1 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.trans22_2 = nn.ConvTranspose2d(256, 32, 2, 2, 0)
        self.ident22_3 = Identical_Block(288, 64, 1, 1, 0)
        self.block22_4 = Backbone_Block(96, 32, 3, 1, 1)

        self.trans12_1 = nn.ConvTranspose2d(32, 16, 2, 2, 0)
        self.trans12_2 = nn.ConvTranspose2d(64, 16, 2, 2, 0)
        self.ident12_3 = Identical_Block(80, 32, 1, 1, 0)
        self.block12_4 = Backbone_Block(48, 16, 3, 1, 1)

        self.trans02_1 = nn.ConvTranspose2d(16, 16, 2, 2, 0)
        self.ident02_2 = Identical_Block(16, 16, 1, 1, 0)
        self.block02_3 = Backbone_Block(16, 16, 3, 1, 1)

        self.last = Identical_Block(16, 16, 1, 1, 0)

        self.output1 = nn.Conv2d(16, class_num, 1, 1, 0)
        self.output2 = nn.Softmax()

    def encoder(self, x, context=None, skip=None):
        x = x[:,1,...] if context else x[:,0,...]
        x00 = self.layer00(x)
        if skip == 1:
            return x00
        x01 = self.layer01(x00)        
        x10 = self.layer10(x01)
        x11 = self.layer11(x10)
        if skip == 2:
            return x11         
        x20 = self.layer20(x11)
        x21 = self.layer21(x20)
        if skip == 3:
            return x21
        x30 = self.layer30(x21)
        x31 = self.layer31(x30)
        if skip == 4:
            return x31        
        x40 = self.layer40(x31)
        x41 = self.layer41(x40)
        return x41

    def forward(self, x):
        x51 = self.encoder(x)
        y51 = self.encoder(x, context=True)
        y52 = self.trans52(y51[:, :, y51.shape[2]//2-4:y51.shape[2]//2+4, y51.shape[3]//2-4: y51.shape[3]//2+4])
        x53 = self.ident53(torch.cat([x51, y52], dim=1))

        x42_1 = self.trans42_1(x53)
        y41_1 = self.encoder(x, skip=4)
        y42_1 = self.encoder(x, context=True, skip=4)
        y42_2 = self.trans42_2(y42_1[:, :, y42_1.shape[2]//2-8:y42_1.shape[2]//2+8, y42_1.shape[3]//2-8: y42_1.shape[3]//2+8])
        y43_1 = self.ident42_3(torch.cat([y41_1, y42_2], dim=1))
        x43_2 = self.block42_4(torch.cat([x42_1, y43_1], dim=1))

        x32_1 = self.trans32_1(x43_2)
        y31_1 = self.encoder(x, skip=3)
        y32_1 = self.encoder(x, context=True, skip=3)
        y32_2 = self.trans32_2(y32_1[:, :, y32_1.shape[2]//2-16:y32_1.shape[2]//2+16, y32_1.shape[3]//2-16: y32_1.shape[3]//2+16])
        y33_1 = self.ident32_3(torch.cat([y31_1, y32_2], dim=1))
        x33_2 = self.block32_4(torch.cat([x32_1, y33_1], dim=1))

        x22_1 = self.trans22_1(x33_2)
        y21_1 = self.encoder(x, skip=2)
        y22_1 = self.encoder(x, context=True, skip=2)
        y22_2 = self.trans22_2(y22_1[:, :, y22_1.shape[2]//2-32:y22_1.shape[2]//2+32, y22_1.shape[3]//2-32: y22_1.shape[3]//2+32])
        y23_1 = self.ident22_3(torch.cat([y21_1, y22_2], dim=1))
        x23_2 = self.block22_4(torch.cat([x22_1, y23_1], dim=1))

        x12_1 = self.trans12_1(x23_2)
        y11_1 = self.encoder(x, skip=1)
        y12_1 = self.encoder(x, context=True, skip=1)
        y12_2 = self.trans12_2(y12_1[:, :, y12_1.shape[2]//2-64:y12_1.shape[2]//2+64, y12_1.shape[3]//2-64: y12_1.shape[3]//2+64])
        y13_1 = self.ident12_3(torch.cat([y11_1, y12_2], dim=1))
        x13_2 = self.block12_4(torch.cat([x12_1, y13_1], dim=1))

        x01 = self.trans02_1(x13_2)
        x02 = self.ident02_2(x01)
        x03 = self.block02_3(x02)

        last = self.last(x03)

        output1 = self.output1(last)
        output2 = self.output2(output1)
        return output2

def mrn_se_resnext101_32x4d(class_num):
    return MRN(class_num)