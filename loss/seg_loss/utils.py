# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> utils
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/04/2021 20:12
=================================================='''

import torch
import numpy as np


def make_one_hot(input, num_classes=None):
    if num_classes is None:
        num_classes = input.max() + 1
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu().long, 1)
    return result
