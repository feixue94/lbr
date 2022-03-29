# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 下午2:38
# @Author  : Fei Xue
# @Email   : fx221@cam.ac.uk
# @File    : segloss.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.seg_loss.crossentropy_loss import CrossEntropy


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


class SegLoss(nn.Module):
    def __init__(self, segloss_name, use_cls=None, use_hiera=None, use_seg=None,
                 cls_weight=1., hiera_weight=1., label_weights=None):
        super(SegLoss, self).__init__()

        if use_seg:
            if segloss_name == 'ce':
                self.seg_loss = CrossEntropy(weights=label_weights)
            elif segloss_name == 'ceohem':
                self.seg_loss = CrossEntropyLossWithOHEM(ohem_ratio=0.7, weight=label_weights)
            elif segloss_name == 'sceohem':
                self.seg_loss = SoftCrossEntropyLossWithOHEM(ohem_ratio=0.7)
        else:
            self.seg_loss = None
        if use_cls:
            self.cls_loss = nn.BCEWithLogitsLoss(weight=label_weights)
            self.cls_weight = cls_weight
        else:
            self.cls_loss = None

        if use_hiera:
            # self.cls_hiera = nn.BCEWithLogitsLoss(weight=label_weights)
            self.cls_hiera = nn.CrossEntropyLoss(weight=label_weights)
            self.hiera_weight = hiera_weight
        else:
            self.cls_hiera = None

    def forward(self, pred_seg=None, gt_seg=None, pred_cls=None, gt_cls=None, pred_hiera=None, gt_hiera=None):
        total_loss = 0
        output = {

        }
        if self.seg_loss is not None:
            seg_error = self.seg_loss(pred_seg, gt_seg)
            total_loss = total_loss + seg_error
            output["seg_loss"] = seg_error

        if self.cls_loss is not None:
            cls_error = self.cls_loss(pred_cls, gt_cls)
            total_loss = total_loss + cls_error * self.cls_weight
            output["cls_loss"] = cls_error

        if self.cls_hiera is not None:
            # print (pred_hiera.shape, gt_hiera.shape)
            hiera_error = self.cls_hiera(pred_hiera, torch.argmax(gt_hiera, 1).long())
            total_loss = total_loss + hiera_error * self.hiera_weight
            output["hiera_loss"] = hiera_error

        output["loss"] = total_loss
        return output
