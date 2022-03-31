# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/15 下午7:26
@Auth ： Fei Xue
@File ： layers.py
@Email： feixue@pku.edu.cn
"""

import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)
