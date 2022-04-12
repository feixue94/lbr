# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> pspnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   2021-05-12 19:22
=================================================='''

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.pspnet.decoder import PSPDecoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.base.modules import Flatten, Activation, Conv2dReLU
import segmentation_models_pytorch.base.initialization as init
from loss.seg_loss.crossentropy_loss import cross_entropy2d

from typing import Optional, Union


class CondLayer(nn.Module):
    """
    implementation of the element-wise linear modulation layer
    """

    def __init__(self):
        super(CondLayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, gammas, betas):
        return self.relu((x * gammas.expand_as(x)) + betas.expand_as(x))


def conv(in_planes, out_planes, kernel_size=3, stride=1, bn=False):
    if not bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_planes, affine=True),
            nn.ReLU(inplace=True)
        )


def conv1x1(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
    )


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNetF(SegmentationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_depth: int = 3,
                 psp_out_channels: int = 512,
                 psp_use_batchnorm: bool = True,
                 psp_dropout: float = 0.2,
                 in_channels: int = 3,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 8,
                 hierarchical: bool = False,
                 classification: bool = False,
                 segmentation: bool = False,
                 classes=21428,
                 out_indices=(1, 2, 3),
                 require_spp_feats=False
                 ):
        super(PSPNetF, self).__init__()
        self.classification = classification
        self.require_spp_feats = require_spp_feats

        self.init_modules = []

        self.encoder = get_encoder(name=encoder_name,
                                   in_channels=3,
                                   depth=encoder_depth,
                                   weights=encoder_weights)

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,  # 3, 64, 256
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.seghead = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,  # 8 for robotcar, 4 for obs9, 2 for obs6 & obs4
        )

        if classification:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(self.encoder.out_channels[-1], classes, bias=True)
            )

        self.name = "psp-{}".format(encoder_name)
        self.initialize()

    def compute_seg_loss(self, pred_segs, gt_segs, weights=[1.0, 1.0, 1.0, 1.0]):
        # pred_segs = inputs["masks"]
        # gt_segs = outputs["label"]

        seg_loss = 0

        for pseg, gseg in zip(pred_segs, gt_segs):
            # print("pseg, gseg: ", pseg.shape, gseg.shape)
            gseg = gseg.cuda()
            if len(gseg.shape) == 3:
                gseg = gseg.unsqueeze(1)
            if pseg.shape[2] != gseg.shape[2] or pseg.shape[3] != gseg.shape[3]:
                gseg = F.interpolate(gseg.float(), size=(pseg.shape[2], pseg.shape[3]), mode="nearest")

            # seg_loss += cross_entropy_seg(input=pseg, target=gseg)
            seg_loss += cross_entropy2d(input=pseg, target=gseg.long())

        return seg_loss

    def compute_cls_loss(self, pred_cls, gt_cls, method="cel"):
        cls_loss = 0
        for pc, gc in zip(pred_cls, gt_cls):
            gc = gc.cuda()
            cls_loss += torch.nn.functional.binary_cross_entropy_with_logits(pc, gc)
        return cls_loss

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # batch = x.shape[0]
        # device = self.parameters().device
        # print('device: ', device)
        features = self.encoder(x)
        # for v in features:
        #     print(v.shape)
        decode_feat = self.decoder(*features)

        masks = self.seghead(decode_feat)

        # seg_loss = self.compute_seg_loss(pred_segs=[masks], gt_segs=input['label'])

        output = {"masks": [masks]}
        if self.classification:
            cls = self.cls_head(features[-1])
            output["cls"] = [cls]
        output['feats'] = features

        return output

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

    def initialize(self):
        init.initialize_decoder(self.decoder)

        init.initialize_head(self.seghead)
        if self.classification:
            init.initialize_head(self.cls_head)


if __name__ == '__main__':
    net = PSPNetF(
        encoder_name="timm-resnest50d",
        encoder_weights="imagenet",
        # classes=256,
        # clusters=200,
        encoder_depth=4,
        # psp_out_channels=512,
    ).cuda()

    print(net)
    img = torch.ones((4, 3, 256, 256)).cuda()
    out = net(img)
    if "masks" in out.keys():
        masks = out["masks"]
        print(masks[0].shape, masks[1].shape, masks[2].shape)
    if "cls" in out.keys():
        cls = out["cls"]
        print(cls[0].shape, cls[1].shape, cls[2].shape)
    # print (v.shape for v in masks)
    # print (v.shape for v in cls)
