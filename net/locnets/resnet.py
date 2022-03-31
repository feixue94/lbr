# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> resnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/09/2021 22:11
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, groups=32, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, outplanes)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes, stride, groups, dilation)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = conv1x1(outplanes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNetX(nn.Module):
    def __init__(self, encoder_name='resnext101_32x4d', encoder_depth=3, encoder_weights='ssl', outdim=128,
                 freeze_encoder=False):
        super(ResNetX, self).__init__()

        encoder = get_encoder(name=encoder_name,
                              in_channels=3,
                              depth=encoder_depth,
                              weights=encoder_weights)

        if encoder_depth == 3:
            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,  # 2x ds
                encoder.relu,
                encoder.maxpool,
                encoder.layer1,  # 4x ds
                encoder.layer2,  # 8x ds
            )
            c = 512
            self.ds = 8

            self.conv = nn.Sequential(
                nn.Conv2d(c, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        elif encoder_depth == 2:
            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,  # 2x ds
                encoder.relu,
                encoder.maxpool,
                encoder.layer1,  # 4x ds
            )
            c = 256
            self.ds = 4

            self.conv = nn.Sequential(

                ResBlock(inplanes=256, outplanes=256, groups=32),
                ResBlock(inplanes=256, outplanes=256, groups=32),
                ResBlock(inplanes=256, outplanes=256, groups=32),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        elif encoder_depth == 1:
            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,  # 2x ds
                encoder.relu,
                encoder.maxpool,
            )
            c = 64
            self.ds = 2
            # TODO

        if freeze_encoder:
            print("Freeze encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Detector Head.
        self.convPa = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.convPb = torch.nn.Conv2d(256, self.ds * self.ds, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.convDb = torch.nn.Conv2d(256, outdim, kernel_size=1, stride=1, padding=0)

    def det(self, x):
        x = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(len(features))

        # pass through several CNN layers first
        x = self.conv(x)

        # Detector Head.
        cPa = self.convPa(x)
        semi = self.convPb(cPa)
        semi = torch.sigmoid(semi)

        Hc, Wc = semi.size(2), semi.size(3)
        # recover resolution
        semi = semi.permute([0, 2, 3, 1])
        score = semi.view(semi.size(0), Hc, Wc, self.ds, self.ds)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * self.ds, Wc * self.ds).unsqueeze(1)

        # Descriptor Head
        cDa = self.convDa(x)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        return score, desc

    def forward(self, batch):
        b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        score, desc = self.det(x)

        return {
            "score": score,
            "desc": desc
        }
