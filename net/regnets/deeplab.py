# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> deeplab
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   06/09/2021 13:36
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.base.modules import Flatten, Activation, Conv2dReLU
import segmentation_models_pytorch as smp
from typing import Optional, Union


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabV3Plus(nn.Module):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_output_stride: int = 8,
                 decoder_channels: int = 256,
                 decoder_atrous_rates: tuple = (12, 24, 36),
                 in_channels: int = 3,
                 encoder_depth: int = 3,
                 classes: int = 1,
                 upsampling: int = 8,
                 activation: Optional[str] = None,
                 classification: bool = False,
                 ):
        super(DeepLabV3Plus, self).__init__()
        self.classification = classification

        self.encoder = get_encoder(name=encoder_name,
                                   in_channels=in_channels,
                                   depth=encoder_depth,
                                   weights=encoder_weights,
                                   )

        if encoder_output_stride == 8:
            self.encoder.make_dilated(
                stage_list=[4, 5],
                dilation_list=[2, 4]
            )

        elif encoder_output_stride == 16:
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )
        else:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.seghead = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if self.classification:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(self.encoder.out_channels[-1], classes, bias=True)
            )

    def forward(self, x):
        features = self.encoder(x)
        # print('len: ', len(features))
        # for i in range(5):
        #     print(i, features[i].shape)
        decoder_output = self.decoder(*features)

        masks = self.seghead(decoder_output)
        # print('mask: ', masks.shape)
        output = {"masks": [masks]}
        if self.classification:
            cls = self.cls_head(features[-1])
            output["cls"] = [cls]

        output['feats'] = features

        return output
