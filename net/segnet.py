# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 上午11:36
# @Author  : Fei Xue
# @Email   : fx221@cam.ac.uk
# @File    : regnet.py
# @Software: PyCharm

from net.regnets.pspnet import PSPNetF
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
    if network == "pspf":
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
