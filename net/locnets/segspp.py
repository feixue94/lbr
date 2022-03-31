# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> segspp
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   15/06/2021 14:55
=================================================='''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegSpp(nn.Module):
    def __init__(self, input_channel=256):
        super(SegSpp, self).__init__()

        self.conv_desc = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1)
        )

        self.conv_det = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
        )

    def det(self, x):
        desc = self.conv_desc(x)
        # desc = F.normalize(desc, p=2, dim=1)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        semi = self.conv_det(x)
        Hc, Wc = semi.size(2), semi.size(3)
        semi = semi.permute([0, 2, 3, 1])
        score = semi.view(semi.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)
        score = torch.sigmoid(score)

        return score, desc

    def forward(self, x):
        desc = self.conv_desc(x)
        desc = F.normalize(desc, p=2, dim=1)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        semi = self.conv_det(x)
        Hc, Wc = semi.size(2), semi.size(3)
        semi = semi.permute([0, 2, 3, 1])
        score = semi.view(semi.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)

        return {'desc': desc,
                'score': score}
