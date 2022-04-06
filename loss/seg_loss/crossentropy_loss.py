# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> crossentropy_loss
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/04/2021 20:19
=================================================='''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _ohem_mask(loss, ohem_ratio):
    with torch.no_grad():
        values, _ = torch.topk(loss.reshape(-1),
                               int(loss.nelement() * ohem_ratio))
        mask = loss >= values[-1]
    return mask.float()


class BCEWithLogitsLossWithOHEM(nn.Module):
    def __init__(self, ohem_ratio=1.0, pos_weight=None, eps=1e-7):
        super(BCEWithLogitsLossWithOHEM, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=pos_weight)
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)

    def set_ohem_ratio(self, ohem_ratio):
        self.ohem_ratio = ohem_ratio


class CrossEntropyLossWithOHEM(nn.Module):
    def __init__(self,
                 ohem_ratio=1.0,
                 weight=None,
                 ignore_index=-100,
                 eps=1e-7):
        super(CrossEntropyLossWithOHEM, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_index,
                                             reduction='none')
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)

    def set_ohem_ratio(self, ohem_ratio):
        self.ohem_ratio = ohem_ratio


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        loss = 1 - (2. * intersection) / (pred.sum() + target.sum() + self.eps)
        return loss


class SoftCrossEntropyLossWithOHEM(nn.Module):
    def __init__(self, ohem_ratio=1.0, eps=1e-7):
        super(SoftCrossEntropyLossWithOHEM, self).__init__()
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        loss = -(pred * target).sum(1)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)

    def set_ohem_ratio(self, ohem_ratio):
        self.ohem_ratio = ohem_ratio


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)

    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


def cross_entropy_seg(input, target):
    idx = target.cpu().long()
    one_hot_key = torch.FloatTensor(input.shape).zero_()
    one_hot_key = one_hot_key.scatter_(1, idx, 1)
    if one_hot_key.device != input.device:
        one_hot_key = one_hot_key.to(input.device)
    log_p = F.log_softmax(input=input, dim=1)
    loss = -(one_hot_key * log_p)

    loss = loss.sum(1)
    # print("ce_loss: ", loss.shape)
    # loss2 = cross_entropy2d(input=input, target=target.squeeze())
    # print("loss: ", loss.mean(), loss2)

    # loss = F.nll_loss(input=log_p, target=target.view(-1), reduction="sum")
    return loss.mean()


class CrossEntropy(nn.Module):
    def __init__(self, weights=None):
        super(CrossEntropy, self).__init__()
        self.weights = weights

    def forward(self, input, target):
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(input.shape).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != input.device:
            one_hot_key = one_hot_key.to(input.device)
        log_p = F.log_softmax(input=input, dim=1)
        loss = -(one_hot_key * log_p)

        if self.weights is not None:
            loss = (loss * self.weights[None, :, None, None].expand_as(loss)).sum(1)  # + self.smooth [B, C, H, W]
        else:
            loss = loss.sum(1)
        # print("ce_loss: ", loss.shape)
        # loss2 = cross_entropy2d(input=input, target=target.squeeze())
        # print("loss: ", loss.mean(), loss2)

        # loss = F.nll_loss(input=log_p, target=target.view(-1), reduction="sum")
        return loss.mean()


if __name__ == '__main__':
    target = torch.randint(0, 4, (1, 1, 4, 4)).cuda()
    input = torch.rand((1, 4, 4, 4)).cuda()
    print("target: ", target.shape)

    net = CrossEntropy().cuda()
    out = net(input, target)
    print(out)
