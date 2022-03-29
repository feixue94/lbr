# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/16 下午3:03
@Auth ： Fei Xue
@File ： semloc_loss.py
@Email： feixue@pku.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemlocLoss:
    def __init__(self, use_feat=False, use_obj=False, use_img=False):
        self.use_feat = use_feat
        self.use_obj = use_obj
        self.use_img = use_img

    def feat_loss(self):
        pass

    def obj_loss(self):
        pass

    def img_loss(self):
        pass

    def __call__(self, inputs):
        pass
