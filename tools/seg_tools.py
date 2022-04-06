# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> seg_tools
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   30/04/2021 15:47
=================================================='''

import numpy as np


def seg_to_rgb(seg, maps):
    """
    Args:
        seg: [C, H, W]
        maps: [id, R, G, B]
    Returns: [H, W, R/G/B]
    """
    pred_label = seg.max(0).cpu().numpy()
    output = np.zeros(shape=(pred_label.shape[0], pred_label.shape[1], 3), dtype=np.uint8)

    for label in maps.keys():
        rgb = maps[label]
        output[pred_label == label] = np.uint8(rgb)
    return output


def label_to_rgb(label, maps):
    output = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for k in maps.keys():
        rgb = maps[k]
        if type(rgb) is int:
            b = rgb % 256
            r = rgb // (256 * 256)
            g = (rgb - r * 256 * 256) // 256
            rgb = np.array([r, g, b], np.uint8)
            # bgr = np.array([b, g, r], np.uint8)
        output[label == k] = np.uint8(rgb)
    return output


def label_to_bgr(label, maps):
    output = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for k in maps.keys():
        rgb = maps[k]
        if type(rgb) is int:
            b = rgb % 256
            r = rgb // (256 * 256)
            g = (rgb - r * 256 * 256) // 256
            # rgb = np.array([r, g, b], np.uint8)
            bgr = np.array([b, g, r], np.uint8)
        output[label == k] = np.uint8(bgr)
    return output


def rgb_to_bgr(img):
    out = np.zeros_like(img)
    out[:, :, 0] = img[:, :, 2]
    out[:, :, 1] = img[:, :, 1]
    out[:, :, 2] = img[:, :, 0]
    return out


def read_seg_map(path):
    map = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip("\n").split(' ')
            map[int(l[1])] = np.array([np.uint8(l[2]), np.uint8(l[3]), np.uint8(l[4])], np.uint8)

    return map


def read_seg_map_with_group(path):
    map = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(" ")
            rid = int(l[0])
            grgb = int(l[1])
            gid = int(l[2])
            if rid in map.keys():
                map[rid][gid] = grgb
            else:
                map[rid] = {}
                map[rid][gid] = grgb
    return map


def read_seg_map_without_group(path):
    map = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(" ")
            grgb = int(l[0])
            gid = int(l[1])
            map[gid] = grgb
    return map


## code is from https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc
