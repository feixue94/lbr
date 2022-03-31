# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> gem
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   22/08/2021 15:34
=================================================='''
import pdb
import numpy as np
import torch
from torch.autograd import Variable

import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class GeneralizedMeanPooling(Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = Parameter(torch.ones(1) * norm)


class Gem(nn.Module):
    def __init__(self, ut_dim=2048, norm_features=False,
                 pooling='gem', gemp=3, center_bias=0,
                 dropout_p=None, without_fc=False, **kwargs):
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias
        self.adpool = GeneralizedMeanPoolingP(norm=gemp)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(2048, out_dim)
        self.fc_name = 'last_linear'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)

        if self.detach:
            # stop the back-propagation here, if needed
            x = Variable(x.detach())
            x = self.id(x)  # fake transformation

        if self.center_bias > 0:
            b = self.center_bias
            bias = 1 + torch.FloatTensor([[[[0, 0, 0, 0], [0, b, b, 0], [0, b, b, 0], [0, 0, 0, 0]]]]).to(x.device)
            bias = torch.nn.functional.interpolate(bias, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = x * bias

        # global pooling
        x = self.adpool(x)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x.squeeze_()
        if not self.without_fc:
            x = self.fc(x)

        x = l2_normalize(x, axis=-1)
        return x
