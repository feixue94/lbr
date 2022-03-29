# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 上午11:36
# @Author  : Fei Xue
# @Email   : fx221@cam.ac.uk
# @File    : regnet.py
# @Software: PyCharm

import torch

import torch
import torch.nn as nn
import segmentation_models_pytorch as sgm
from net.regnets.pspnet import PSPNet, PSPNetH, PSPNetHF, PSPNetC, PSPNetF, PSPNetH2, PSPNetH3
from net.regnets.upernet import UperNet
from net.regnets.deeplab import DeepLabV3Plus


def get_segnet(network,
               n_classes,
               encoder_name=None,
               encoder_weights=None,
               out_channels=512,
               classification=True,
               segmentation=True,
               encoder_depth=4,
               upsampling=8,
               ):
    if network == 'upernet':
        net = UperNet(
            encoder_name='swin',
            channels=out_channels,
            in_channels=[128, 256, 512, 1024],
            num_classes=n_classes,
            in_index=[0, 1, 2, 3],
            classification=classification
        )
    elif network == "pspf":
        net = PSPNetF(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,

            # aux_params=aux_params,
            encoder_depth=encoder_depth,  # 3 for robotcar, 4 for obs 6 & 9
            psp_out_channels=out_channels,
            # hierarchical=hierarchical,
            classification=classification,
            # segmentation=segmentation,
            upsampling=upsampling,
            # classes=21428,
            # classes=3962,
            classes=n_classes,
        )
    elif network == 'deeplabv3p':
        net = DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=out_channels,
            decoder_atrous_rates=(12, 24, 36),
            encoder_output_stride=8,
            encoder_depth=encoder_depth,
            classification=classification,
            upsampling=upsampling,
            classes=n_classes,
        )

    return net


def get_seg_network(network,
                    n_class,
                    encoder_name=None,
                    encoder_weights=None,
                    hierarchical=False,
                    classification=False,
                    segmentation=False,
                    clusters=1,
                    out_channels=512,
                    aux_params=None):
    if network == 'upernet':
        net = UperNet(
            encoder_name='swin',
            # channels=1024,
            channels=out_channels,
            in_channels=[128, 256, 512, 1024],
            num_classes=n_class,
            in_index=[0, 1, 2, 3],
            classification=classification
        )
    elif network == "pspf":
        # """
        # aux_params = dict(
        #     pooling='avg',  # one of 'avg', 'max'
        #     dropout=None,  # dropout ratio, default is None
        #     # activation='sigmoid',  # activation function, default is None
        #     classes=n_class,
        # )
        # net = sgm.PSPNet(
        #     encoder_name="timm-resnest50d",
        #     encoder_weights="imagenet",
        #     classes=n_class,
        #     aux_params=aux_params,
        #     encoder_depth=4,
        #     psp_out_channels=1024,
        #     upsampling=2,
        # )
        net = PSPNetF(
            # encoder_name="timm-resnest50d",
            # encoder_weights="imagenet",
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,

            # aux_params=aux_params,
            encoder_depth=4,  # 3 for robotcar, 4 for obs 6 & 9
            psp_out_channels=out_channels,
            # hierarchical=hierarchical,
            classification=classification,
            # segmentation=segmentation,
            upsampling=8,
            # classes=21428,
            # classes=3962,
            classes=n_class,
        )
        # """
    elif network == "psp":
        net = PSPNet(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            classes=n_class,
            clusters=clusters,
            # aux_params=aux_params,
            encoder_depth=5,
            psp_out_channels=out_channels,
            hierarchical=hierarchical,
            classification=classification,
            segmentation=segmentation,
            upsampling=4,
        )
        # """
    elif network == "psph":
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=None,  # dropout ratio, default is None
            # activation='sigmoid',  # activation function, default is None
            classes=128,
        )
        net = PSPNetH2(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            classes=[170, 128],
            encoder_depth=4,
            classification=classification,
        )
        # net = PSPNetH3(
        #     encoder_name="timm-resnest50d",
        #     encoder_weights="imagenet",
        #     aux_params=aux_params,
        #     encoder_depth=3,
        #     psp_out_channels=256,
        #     classes=128,
        #     upsampling=2,
        # )
    elif network == "psphf":
        net = PSPNetHF(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            # aux_params=aux_params,
            encoder_depth=4,
            psp_out_channels=out_channels,
            # hierarchical=hierarchical,
            classification=classification,
            # segmentation=segmentation,
            upsampling=8,
            classes=[256, 4096, 21428],
        )
    elif network == "pspc":
        net = PSPNetC(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            # aux_params=aux_params,
            encoder_depth=5,
            # psp_out_channels=out_channels,
            # hierarchical=hierarchical,
            # classification=classification,
            # segmentation=segmentation,
            # upsampling=8,
            classes=21428,
        )
    elif network == "unet":
        net = sgm.Unet(
            encoder_name="timm-resnest50d",
            encoder_weights="imagenet",
            classes=n_class,
            aux_params=aux_params
        )
    elif network == "deeplabv3":
        net = sgm.DeepLabV3(
            encoder_name="resnext50_32x4d",
            encoder_weights="imagenet",
            classes=n_class,
            aux_params=aux_params
        )
    elif network == "deeplabv3plus":
        net = sgm.DeepLabV3Plus(
            encoder_name="resnext50_32x4d",
            encoder_weights="imagenet",
            classes=n_class,
            aux_params=aux_params
        )
    elif network == "pan":
        net = sgm.PAN(
            encoder_name="resnext50_32x4d",
            encoder_weights="imagenet",
            classes=n_class,
            aux_params=aux_params
        )
    return net


if __name__ == '__main__':
    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=None,  # dropout ratio, default is None
        # activation='sigmoid',  # activation function, default is None
        classes=299,
    )
    network = "deeplabv3"
    network = "deeplabv3plus"
    network = "pan"
    net = get_seg_network(network=network, n_class=299, aux_params=aux_params)
    print(net)
