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
import torchvision.models as models
from net.swin_transformer.swin_transformer import swin_base
import numpy as np
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


class PSPNetH3(SegmentationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_depth: int = 3,
                 psp_out_channels: int = 512,
                 psp_use_batchnorm: bool = True,
                 psp_dropout: float = 0.2,
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 8,
                 aux_params: Optional[dict] = None,
                 ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        print(self.encoder.out_channels)

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        if aux_params:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "psp-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])

            # return masks, labels
            return {"masks": [masks],
                    "cls": [labels]}

        return {"masks": masks}


class PSPNetV1(SegmentationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_depth: int = 3,
                 psp_out_channels: int = 512,
                 psp_use_batchnorm: bool = True,
                 psp_dropout: float = 0.2,
                 in_channels: int = 3,
                 classes: int = 1,
                 clusters: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 8,
                 hierarchical: bool = False,
                 classification: bool = False
                 ):
        super(PSPNetV1, self).__init__()

        self.init_modules = []

        self.encoder = get_encoder(name=encoder_name, in_channels=3, depth=encoder_depth, weights=encoder_weights)

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.seghead = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        self.init_modules.append(self.seghead)

        if classification:
            self.cls_head2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(512, classes, bias=True)
            )

            self.init_modules.append(self.cls_head2)
        else:
            self.cls_head2 = None

        self.hierarchical = hierarchical
        if self.hierarchical:
            self.cond = CondLayer()
            channels = [28, 256, 512, 1024]
            self.genc1 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                    padding=1)
            self.genc2 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                    padding=1)

            self.gen_gamma_1 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                          use_batchnorm=False, padding=1)
            self.gen_beta_1 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                         use_batchnorm=False, padding=1)

            self.gen_gamma_2 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                          use_batchnorm=False, padding=1)
            self.gen_beta_2 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                         use_batchnorm=False, padding=1)

            self.seg_conv1 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                        padding=1)
            self.seg_conv2 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                        padding=1)

            # self.gencov1 = conv(in_planes=channels[2], out_planes=channels[2], stride=1, bn=False)
            self.cls_head1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(512, clusters, bias=True)
            )

            self.init_modules.append(self.genc1)
            self.init_modules.append(self.genc2)
            self.init_modules.append(self.gen_gamma_1)
            self.init_modules.append(self.gen_beta_1)
            self.init_modules.append(self.gen_gamma_2)
            self.init_modules.append(self.gen_beta_2)
            self.init_modules.append(self.seg_conv1)
            self.init_modules.append(self.seg_conv2)
            self.init_modules.append(self.cls_head1)

            if classification:
                self.cls_conv1 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                            padding=1)
                self.cls_conv2 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3,
                                            padding=1)

                self.init_modules.append(self.cls_conv1)
                self.init_modules.append(self.cls_conv2)

        self.name = "psp-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(features[5].shape)
        # print(len(features))
        # exit(0)

        decoder_output = self.decoder(*features)  # 3, 64, 256, 512
        x_last = features[-1]  # 512
        # print("x_last: ", x_last.shape)

        if self.hierarchical:
            out = self.genc1(x_last)
            out = self.genc2(out)
            labels1 = self.cls_head1(out)

            out_gama1 = self.gen_gamma_1(out)
            out_beta1 = self.gen_beta_1(out)

            out_gama2 = self.gen_gamma_2(out)
            out_beta2 = self.gen_beta_2(out)

            # print (decoder_output.shape, out_gama1.shape, out_beta1.shape)
            seg_out = self.seg_conv1(self.cond(decoder_output, out_gama1, out_beta1))
            seg_out = self.seg_conv2(self.cond(seg_out, out_gama2, out_beta2))

            masks = self.seghead(seg_out)

            if self.cls_head2 is not None:
                cls_out = self.cond(self.cls_conv1(x_last), out_gama1, out_beta1)
                cls_out = self.cond(self.cls_conv2(cls_out), out_gama2, out_beta2)
                labels2 = self.cls_head2(cls_out)

                return masks, labels2, labels1
            else:
                return masks, labels1
        else:
            masks = self.seghead(decoder_output)

            if self.cls_head2 is not None:
                labels = self.cls_head2(features[-1])
                return masks, labels

            else:
                return masks

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
        for m in self.init_modules:
            init.initialize_head(m)
        # init.initialize_decoder(self.decoder)
        # init.initialize_head(self.seg_head)
        # if self.classification_head is not None:
        #     init.initialize_head(self.classification_head)


class PSPNet(SegmentationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_depth: int = 3,
                 psp_out_channels: int = 512,
                 psp_use_batchnorm: bool = True,
                 psp_dropout: float = 0.2,
                 in_channels: int = 3,
                 classes: int = 1,
                 clusters: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 8,
                 hierarchical: bool = False,
                 classification: bool = False,
                 segmentation: bool = False,
                 ):
        super(PSPNet, self).__init__()
        self.segmentation = segmentation

        self.init_modules = []

        self.encoder = get_encoder(name=encoder_name, in_channels=3,
                                   depth=encoder_depth,
                                   weights=encoder_weights)

        if self.segmentation:
            self.decoder = PSPDecoder(
                encoder_channels=self.encoder.out_channels[0:4],
                # encoder_channels=self.encoder.out_channels,
                use_batchnorm=psp_use_batchnorm,
                out_channels=psp_out_channels,
                dropout=psp_dropout,
            )

            self.seghead = SegmentationHead(
                in_channels=psp_out_channels,
                out_channels=classes,
                kernel_size=3,
                activation=activation,
                upsampling=upsampling,
            )

            self.init_modules.append(self.seghead)

            if classification:
                self.cls_head2 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    nn.Linear(512, classes, bias=True)
                )

                self.init_modules.append(self.cls_head2)
            else:
                self.cls_head2 = None

        self.hierarchical = hierarchical
        if self.hierarchical:
            self.cls_head1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(1024, clusters, bias=True)
            )
            self.init_modules.append(self.cls_head1)

            if self.segmentation:
                self.cond = CondLayer()
                channels = [128, 256, 512, 1024, 2048]
                # self.genc1 = Conv2dReLU(in_channels=channels[3], out_channels=channels[2], stride=1, kernel_size=3, padding=1)
                # self.genc2 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=3, padding=1)

                self.gen_gamma_1 = conv1x1(channels[3], channels[2])
                self.gen_beta_1 = conv1x1(channels[3], channels[2])

                self.gen_gamma_2 = conv1x1(channels[3], channels[2])
                self.gen_beta_2 = conv1x1(channels[3], channels[2])

                self.seg_conv1 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=1,
                                            padding=0, use_batchnorm=True)
                self.seg_conv2 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1, kernel_size=1,
                                            padding=0, use_batchnorm=True)

                self.init_modules.append(self.gen_gamma_1)
                self.init_modules.append(self.gen_beta_1)
                self.init_modules.append(self.gen_gamma_2)
                self.init_modules.append(self.gen_beta_2)
                self.init_modules.append(self.seg_conv1)
                self.init_modules.append(self.seg_conv2)

                if classification:
                    self.cls_conv1 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1,
                                                kernel_size=1,
                                                padding=0, use_batchnorm=True)
                    self.cls_conv2 = Conv2dReLU(in_channels=channels[2], out_channels=channels[2], stride=1,
                                                kernel_size=1,
                                                padding=0, use_batchnorm=True)
                    #
                    self.init_modules.append(self.cls_conv1)
                    self.init_modules.append(self.cls_conv2)

        self.name = "psp-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # batch = x.shape[0]
        features = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(features[5].shape)
        # print(len(features))
        # print (v.shape for v in features)
        # exit(0)

        if self.hierarchical:
            x_last = features[-1]  # 1024
            # out = self.genc1(x_last)
            # out = self.genc2(out)
            # print("x_last: ", x_last.shape)
            labels1 = self.cls_head1(x_last)

            if self.segmentation:
                decoder_output = self.decoder(*features[0:4])  # 3, 64, 256, 512
                out_gama1 = self.gen_gamma_1(x_last)
                out_beta1 = self.gen_beta_1(x_last)
                out_gama1 = F.adaptive_avg_pool2d(out_gama1, output_size=1)
                out_beta1 = F.adaptive_avg_pool2d(out_beta1, output_size=1)

                out_gama2 = self.gen_gamma_2(x_last)
                out_beta2 = self.gen_beta_2(x_last)
                out_gama2 = F.adaptive_avg_pool2d(out_gama2, output_size=1)
                out_beta2 = F.adaptive_avg_pool2d(out_beta2, output_size=1)

                # print (decoder_output.shape, out_gama1.shape, out_beta1.shape)
                seg_out = self.cond(self.seg_conv1(decoder_output), out_gama1, out_beta1)
                seg_out = self.cond(self.seg_conv2(seg_out), out_gama2, out_beta2)

                masks = self.seghead(seg_out)

                if self.cls_head2 is not None:
                    x_second_last = features[-2]
                    cls_out = self.cond(self.cls_conv1(x_second_last), out_gama1, out_beta1)
                    cls_out = self.cond(self.cls_conv2(cls_out), out_gama2, out_beta2)
                    labels2 = self.cls_head2(cls_out)

                    return masks, labels2, labels1
                else:
                    return masks, None, labels1
            else:
                return None, None, labels1
        else:
            if self.segmentation:
                decoder_output = self.decoder(*features)
                masks = self.seghead(decoder_output)

                if self.cls_head2 is not None:
                    labels = self.cls_head2(features[-1])
                    return masks, labels

                else:
                    return masks

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
        if self.segmentation:
            init.initialize_decoder(self.decoder)
        for m in self.init_modules:
            init.initialize_head(m)
        # init.initialize_decoder(self.decoder)
        # init.initialize_head(self.seg_head)
        # if self.classification_head is not None:
        #     init.initialize_head(self.classification_head)


class PSPNetH(SegmentationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_depth: int = 3,
                 psp_out_channels: int = 512,
                 psp_use_batchnorm: bool = True,
                 psp_dropout: float = 0.2,
                 in_channels: int = 3,
                 classes: int = 1,
                 clusters: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 8,
                 hierarchical: bool = False,
                 classification: bool = False,
                 segmentation: bool = False,
                 ):
        super(PSPNetH, self).__init__()
        self.classification = classification

        self.init_modules = []

        self.encoder = get_encoder(name=encoder_name, in_channels=3,
                                   depth=encoder_depth,
                                   weights=encoder_weights)
        self.decoder_l1 = PSPDecoder(
            encoder_channels=self.encoder.out_channels[0:3],  # 3, 64, 256
            use_batchnorm=psp_use_batchnorm,
            out_channels=256,
            dropout=psp_dropout,
        )

        self.decoder_l2 = PSPDecoder(
            encoder_channels=self.encoder.out_channels[0:4],  # 3, 64, 256, 512
            use_batchnorm=psp_use_batchnorm,
            out_channels=512,
            dropout=psp_dropout,
        )

        self.decoder_l3 = PSPDecoder(
            encoder_channels=self.encoder.out_channels[0:5],  # 3, 64, 256, 1024
            use_batchnorm=psp_use_batchnorm,
            out_channels=1024,
            dropout=psp_dropout,
        )

        self.seghead1 = SegmentationHead(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            activation=activation,
            upsampling=4,
        )

        self.seghead2 = SegmentationHead(
            in_channels=512,
            out_channels=32,
            kernel_size=3,
            activation=activation,
            upsampling=8,
        )

        self.seghead3 = SegmentationHead(
            in_channels=1024,
            out_channels=16,
            kernel_size=3,
            activation=activation,
            upsampling=16,
        )

        if classification:
            self.cls_head1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(256, 64, bias=True)
            )

            self.cls_head2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(512, 32, bias=True)
            )

            self.cls_head3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(1024, 16, bias=True)
            )
        self.name = "psp-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # batch = x.shape[0]
        features = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(features[5].shape)
        # print(len(features))
        # print (v.shape for v in features)
        # exit(0)

        features_l1 = [features[0], features[1], features[2]]
        decode_feat_l1 = self.decoder_l1(*features_l1)
        features_l2 = [features[0], features[1], decode_feat_l1, features[3]]
        decode_feat_l2 = self.decoder_l2(*features_l2)
        features_l3 = [features[0], features[1], decode_feat_l1, decode_feat_l2, features[4]]
        decode_feat_l3 = self.decoder_l3(*features_l3)

        masks1 = self.seghead1(decode_feat_l1)
        masks2 = self.seghead2(decode_feat_l2)
        masks3 = self.seghead3(decode_feat_l3)

        if self.classification:
            cls1 = self.cls_head1(features[2])
            cls2 = self.cls_head2(features[3])
            cls3 = self.cls_head3(features[4])

            return {"masks": [masks1, masks2, masks3],
                    "cls": [cls1, cls2, cls3]}
        else:
            return {"masks": [masks1, masks2, masks3]}

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
        init.initialize_decoder(self.decoder_l1)
        init.initialize_decoder(self.decoder_l2)
        init.initialize_decoder(self.decoder_l3)

        init.initialize_head(self.seghead1)
        init.initialize_head(self.seghead2)
        init.initialize_head(self.seghead3)
        if self.classification:
            init.initialize_head(self.cls_head1)
            init.initialize_head(self.cls_head2)
            init.initialize_head(self.cls_head3)


class PSPNetH2(SegmentationModel):
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
                 classes=[169, 128],
                 ):
        super(PSPNetH2, self).__init__()
        self.classification = classification
        # out_channels = [256, 512, 1024]
        out_channels = [1024, 256, 1024]
        # n_classes = [256, 256 * 16, 256 * 16 * 8]
        n_classes = classes

        self.init_modules = []

        self.encoder = get_encoder(name=encoder_name, in_channels=3,
                                   depth=encoder_depth,
                                   weights=encoder_weights)

        self.decoder_l1 = PSPDecoder(
            encoder_channels=self.encoder.out_channels[3:5],  # 3, 64, 256, 512
            use_batchnorm=psp_use_batchnorm,
            out_channels=out_channels[0],
            dropout=psp_dropout,
        )

        self.decoder_l2 = PSPDecoder(
            encoder_channels=self.encoder.out_channels[0:3],  # 3, 64, 256
            use_batchnorm=psp_use_batchnorm,
            out_channels=out_channels[1],
            dropout=psp_dropout,
        )

        self.seghead1 = SegmentationHead(
            in_channels=out_channels[0],
            out_channels=n_classes[0],
            kernel_size=3,
            activation=activation,
            upsampling=4,
        )

        self.seghead2 = SegmentationHead(
            in_channels=out_channels[1] + classes[0],
            # in_channels=out_channels[1],
            out_channels=classes[1],
            kernel_size=3,
            activation=activation,
            upsampling=1,
        )

        self.deconv = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=classes[0], out_channels=classes[1], kernel_size=3, stride=2, padding=1,
            #                    output_padding=1),
            # nn.BatchNorm2d(classes[1], affine=True),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=classes[1], out_channels=classes[1], kernel_size=3, stride=2, padding=1,
            #                    output_padding=1),
            # nn.BatchNorm2d(classes[1], affine=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=classes[0], out_channels=classes[0], kernel_size=1, padding=0),
        )

        if classification:
            self.cls_head1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(out_channels[0], n_classes[0], bias=True)
            )

            self.cls_head2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(out_channels[1], n_classes[1], bias=True)
            )
        self.name = "psp-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # batch = x.shape[0]
        features = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(features[5].shape)
        # print(len(features))
        # print (v.shape for v in features)
        # exit(0)

        # features_l1 = [features[0], features[3], features[4]]
        features_l1 = [features[3], features[4]]
        decode_feat_l1 = self.decoder_l1(*features_l1)
        features_l2 = [features[0], features[1], features[2]]
        decode_feat_l2 = self.decoder_l2(*features_l2)

        masks1 = self.seghead1(decode_feat_l1)

        masks1_up = self.deconv(masks1)
        # print("mask1_up: ", masks1_up.shape, decode_feat_l2.shape)
        decode_feat_l2 = torch.cat([masks1_up, decode_feat_l2], dim=1)
        masks2 = self.seghead2(decode_feat_l2)
        # print("mask1: ", masks1.shape, masks2.shape)
        # masks1_rs = F.interpolate(masks1, size=(masks1.shape[2] // 2, masks1.shape[3] // 2), mode="bilinear")
        # masks2_rs = F.interpolate(masks2, size=(masks2.shape[2] // 2, masks2.shape[3] // 2), mode="bilinear")
        # masks3 = torch.repeat_interleave(masks1_rs, dim=1, repeats=masks2_rs.shape[1]) * masks2_rs.repeat(1, masks1_rs.shape[1], 1,
        masks3 = torch.repeat_interleave(masks1, dim=1, repeats=masks2.shape[1]) * masks2.repeat(1, masks1.shape[1], 1,
                                                                                                 1)

        if self.classification:
            cls1 = self.cls_head1(features[4])
            cls2 = self.cls_head2(features[2])
            cls3 = torch.repeat_interleave(cls1, dim=1, repeats=cls2.shape[1]) * cls2.repeat(1, cls1.shape[1])

            return {"masks": [masks3, masks2, masks1],
                    "cls": [cls3, cls2, cls1]}

            # return {"masks": [masks2],
            #         "cls": [cls2]}
        else:
            return {"masks": [masks3, masks2, masks1]}

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
        init.initialize_decoder(self.decoder_l1)
        init.initialize_decoder(self.decoder_l2)

        init.initialize_head(self.seghead1)
        init.initialize_head(self.seghead2)
        if self.classification:
            init.initialize_head(self.cls_head1)
            init.initialize_head(self.cls_head2)


class PSPNetHF(SegmentationModel):
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
                 classes=[256, 4096, 21428],
                 ):
        super(PSPNetHF, self).__init__()
        self.classification = classification
        out_channels = [256, 512, 512]
        # n_classes = [256, 256 * 16, 256 * 16 * 8]
        n_classes = classes

        self.init_modules = []

        self.encoder = get_encoder(name=encoder_name, in_channels=3,
                                   depth=encoder_depth,
                                   weights=encoder_weights)

        self.decoder_l1 = PSPDecoder(
            # encoder_channels=self.encoder.out_channels,  # 3, 64, 256
            encoder_channels=[512, 1024],  # 3, 64, 256
            use_batchnorm=psp_use_batchnorm,
            out_channels=out_channels[0],
            dropout=psp_dropout,
        )

        self.decoder_l2 = PSPDecoder(
            # encoder_channels=self.encoder.out_channels[0:4],  # 3, 64, 256, 512
            encoder_channels=[64, 256, 256],  # 3, 64, 256, 512
            use_batchnorm=psp_use_batchnorm,
            out_channels=out_channels[1],
            dropout=psp_dropout,
        )

        self.decoder_l3 = PSPDecoder(
            # encoder_channels=self.encoder.out_channels[0:5],  # 3, 64, 256, 1024
            encoder_channels=[3, 64, 256, 512],  # 3, 64, 256, 1024
            use_batchnorm=psp_use_batchnorm,
            out_channels=out_channels[2],
            dropout=psp_dropout,
        )

        self.seghead1 = SegmentationHead(
            in_channels=out_channels[0],
            out_channels=n_classes[0],
            kernel_size=3,
            activation=activation,
            upsampling=3,
        )

        self.seghead2 = SegmentationHead(
            in_channels=out_channels[1],
            out_channels=n_classes[1],
            kernel_size=3,
            activation=activation,
            upsampling=2,
        )

        self.seghead3 = SegmentationHead(
            in_channels=out_channels[2],
            out_channels=n_classes[2],
            kernel_size=3,
            activation=activation,
            upsampling=2,
        )

        if classification:
            self.cls_head1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(out_channels[0], n_classes[0], bias=True)
            )

            self.cls_head2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(out_channels[1], n_classes[1], bias=True)
            )

            self.cls_head3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(out_channels[2], n_classes[2], bias=True)
            )
        self.name = "psp-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # batch = x.shape[0]
        features = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(features[5].shape)
        # print(len(features))
        # print (v.shape for v in features)
        # exit(0)

        # features_l1 = [features[0], features[1], features[2]]
        features_l1 = [features[3], features[4]]
        decode_feat_l1 = self.decoder_l1(*features_l1)
        # features_l2 = [features[0], features[1], decode_feat_l1, features[3]]
        features_l2 = [features[1], features[2], decode_feat_l1]
        decode_feat_l2 = self.decoder_l2(*features_l2)
        features_l3 = [features[0], features[1], decode_feat_l1, decode_feat_l2]
        decode_feat_l3 = self.decoder_l3(*features_l3)

        masks1 = self.seghead1(decode_feat_l1)
        masks2 = self.seghead2(decode_feat_l2)
        masks3 = self.seghead3(decode_feat_l3)

        # print (masks1.shape, masks2.shape, masks3.shape)

        if self.classification:
            cls1 = self.cls_head1(decode_feat_l1)
            cls2 = self.cls_head2(decode_feat_l2)
            cls3 = self.cls_head3(decode_feat_l3)

            return {"masks": [masks3, masks2, masks1],
                    "cls": [cls3, cls2, cls1]}
        else:
            return {"masks": [masks3, masks2, masks1]}

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
        init.initialize_decoder(self.decoder_l1)
        init.initialize_decoder(self.decoder_l2)
        init.initialize_decoder(self.decoder_l3)

        init.initialize_head(self.seghead1)
        init.initialize_head(self.seghead2)
        init.initialize_head(self.seghead3)
        if self.classification:
            init.initialize_head(self.cls_head1)
            init.initialize_head(self.cls_head2)
            init.initialize_head(self.cls_head3)


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

        if encoder_name == 'swin':
            self.encoder = swin_base(pretrained=True, out_indices=out_indices)
        else:
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


class PSPNetC(nn.Module):
    def __init__(self, encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_depth: int = 5,
                 classes: int = 21428):
        super(PSPNetC, self).__init__()
        # self.encoder = get_encoder(name=encoder_name, in_channels=3,
        #                            depth=encoder_depth,
        #                            weights=encoder_weights)
        # self.encoder = models.resnext50_32x4d(pretrained=True)
        self.encoder = models.resnext101_32x8d(pretrained=True)
        self.encoder.fc = nn.Linear(2048, classes)

        # self.cls_head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     Flatten(),
        #     nn.Linear(2048, classes, bias=True)
        # )

    def forward(self, x):
        out = self.encoder(x)
        # print(features[-1].shape)
        # out = self.cls_head(features[-1])
        # print (out.shape)
        return {"cls": [out]}


if __name__ == '__main__':
    # net = PSPNet(
    #     encoder_name="timm-resnest50d",
    #     encoder_weights="imagenet",
    #     classes=256,
    #     clusters=200,
    #     encoder_depth=4,
    #     psp_out_channels=512,
    #     hierarchical=True,
    #     classification=True,
    # ).cuda()
    # net = PSPNetH(
    #     encoder_name="timm-resnest50d",
    #     encoder_weights="imagenet",
    # encoder_depth=4,
    # classification=True,
    # ).cuda()

    # net = PSPNetHF(
    #     encoder_name="timm-resnest50d",
    #     encoder_weights="imagenet",
    #     classes=256,
    #     clusters=200,
    # encoder_depth=4,
    # psp_out_channels=512,
    # classification=True,
    # ).cuda()

    # net = PSPNetC(
    #     encoder_name="timm-resnest50d",
    #     encoder_weights="imagenet",
    #     # classes=256,
    #     # clusters=200,
    #     encoder_depth=5,
    #     # psp_out_channels=512,
    # ).cuda()
    net = PSPNetH2(
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
