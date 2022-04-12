# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> gem
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   25/08/2021 21:00
=================================================='''
import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F


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


def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class GEM(nn.Module):
    def __init__(self, in_dim=1024, out_dim=2048, norm_features=False, pooling='gem', gemp=3, center_bias=0,
                 dropout_p=None, without_fc=False, projection=False, cls=False, n_classes=452):
        super(GEM, self).__init__()
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias
        self.projection = projection
        self.cls = cls
        self.n_classes = n_classes

        if self.projection:
            self.proj = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, bias=False),
                # nn.BatchNorm2d(in_dim),
                # nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, bias=False),
            )

        if self.cls:
            self.cls_head = nn.Linear(in_features=in_dim, out_features=self.n_classes)

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling.startswith('gem'):
            self.adpool = GeneralizedMeanPoolingP(norm=gemp)
        else:
            raise ValueError(pooling)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        # self.fc = nn.Linear(512 * block.expansion, out_dim)
        # self.fc = nn.Linear(in_dim, out_dim)
        # self.fc_name = 'fc'
        # self.feat_dim = out_dim

    def forward(self, feat, atten=None):
        if atten is not None:
            with torch.no_grad():
                if atten.shape[2] != feat.shape[2] or atten.shape[3] != feat.shape[3]:
                    atten = F.interpolate(atten, size=(feat.shape[2], feat.shape[3]), mode='bilinear')
                feat = feat * atten.expand_as(feat)

        if self.projection:
            x = self.proj(feat)

        bs, _, H, W = x.shape
        if self.dropout is not None:
            x = self.dropout(x)

        if self.center_bias > 0:
            b = self.center_bias
            bias = 1 + torch.FloatTensor([[[[0, 0, 0, 0], [0, b, b, 0], [0, b, b, 0], [0, 0, 0, 0]]]]).to(x.device)
            bias_resize = torch.nn.functional.interpolate(bias, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = x * bias_resize
        # global pooling
        x = self.adpool(x)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x = x.view(x.shape[0], -1)
        # if not self.without_fc:
        #     x = self.fc(x)
        if self.cls:
            cls_feat = self.cls_head(x)
        x = l2_normalize(x, axis=-1)

        output = {
            'feat': x,
        }

        if self.cls:
            output['cls'] = cls_feat
        return output
