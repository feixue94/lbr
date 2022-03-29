# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> featloss
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   15/06/2021 15:05
=================================================='''

import torch
import torch.nn.functional as F


def feature_loss(pred_desc, gt_desc, pred_conf, gt_conf, weight_desc=1.0, weight_conf=1.0):
    det_loss = F.binary_cross_entropy(pred_conf, gt_conf).mean()
    desc_loss = torch.sum((pred_desc - gt_desc) ** 2, dim=1).mean()

    return desc_loss, det_loss, det_loss * weight_conf + desc_loss * weight_desc
