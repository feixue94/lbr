# -*- coding: utf-8 -*-
"""
@Time ： 2021/2/22 下午2:23
@Auth ： Fei Xue
@File ： extract_sgd2.py
@Email： xuefei@sensetime.com
"""

import os, pdb
import scipy.io
import scipy.misc
import imageio
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class EmptyTensorError(Exception):
    pass


class NoGradientError(Exception):
    pass


def interpolate_dense_features(pos, dense_features, return_corners=False):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    _, h, w = dense_features.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
        w_top_left * dense_features[:, i_top_left, j_top_left] +
        w_top_right * dense_features[:, i_top_right, j_top_right] +
        w_bottom_left * dense_features[:, i_bottom_left, j_bottom_left] +
        w_bottom_right * dense_features[:, i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    if not return_corners:
        return [descriptors, pos, ids]
    else:
        corners = torch.stack([
            torch.stack([i_top_left, j_top_left], dim=0),
            torch.stack([i_top_right, j_top_right], dim=0),
            torch.stack([i_bottom_left, j_bottom_left], dim=0),
            torch.stack([i_bottom_right, j_bottom_right], dim=0)
        ], dim=0)
        return [descriptors, pos, ids, corners]


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos

def process_multiscale(image, model, scales=[.5, 1, 2]):
    b, _, h_init, w_init = image.size()
    device = image.device
    assert (b == 1)

    all_keypoints = torch.zeros([3, 0])
    all_descriptors = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    all_scores = torch.zeros(0)

    previous_dense_features = None
    banned = None
    for idx, scale in enumerate(scales):
        current_image = F.interpolate(
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        _, _, h_level, w_level = current_image.size()

        dense_features = model.dense_feature_extraction(current_image)
        del current_image

        _, _, h, w = dense_features.size()

        # Sum the feature maps.
        if previous_dense_features is not None:
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features

        # Recover detections.
        detections = model.detection(dense_features)
        if banned is not None:
            banned = F.interpolate(banned.float(), size=[h, w]).bool()
            detections = torch.min(detections, ~banned)
            banned = torch.max(
                torch.max(detections, dim=1)[0].unsqueeze(1), banned
            )
        else:
            banned = torch.max(detections, dim=1)[0].unsqueeze(1)
        fmap_pos = torch.nonzero(detections[0].cpu()).t()
        del detections

        # Recover displacements.
        displacements = model.localization(dense_features)[0].cpu()
        displacements_i = displacements[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        displacements_j = displacements[
            1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        del displacements

        mask = torch.min(
            torch.abs(displacements_i) < 0.5,
            torch.abs(displacements_j) < 0.5
        )
        fmap_pos = fmap_pos[:, mask]
        valid_displacements = torch.stack([
            displacements_i[mask],
            displacements_j[mask]
        ], dim=0)
        del mask, displacements_i, displacements_j

        fmap_keypoints = fmap_pos[1:, :].float() + valid_displacements
        del valid_displacements

        try:
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device),
                dense_features[0]
            )
        except EmptyTensorError:
            continue
        fmap_pos = fmap_pos[:, ids]
        fmap_keypoints = fmap_keypoints[:, ids]
        del ids

        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        del fmap_keypoints

        descriptors = F.normalize(raw_descriptors, dim=0).cpu()
        del raw_descriptors

        keypoints[0, :] *= h_init / h_level
        keypoints[1, :] *= w_init / w_level

        fmap_pos = fmap_pos.cpu()
        keypoints = keypoints.cpu()

        keypoints = torch.cat([
            keypoints,
            torch.ones([1, keypoints.size(1)]) * 1 / scale,
        ], dim=0)

        scores = dense_features[
                     0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
                 ].cpu() / (idx + 1)
        del fmap_pos

        all_keypoints = torch.cat([all_keypoints, keypoints], dim=1)
        all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
        all_scores = torch.cat([all_scores, scores], dim=0)
        del keypoints, descriptors

        previous_dense_features = dense_features
        del dense_features
    del previous_dense_features, banned

    keypoints = all_keypoints.t().numpy()
    del all_keypoints
    scores = all_scores.numpy()
    del all_scores
    descriptors = all_descriptors.t().numpy()
    del all_descriptors
    return keypoints, scores, descriptors


def preprocess_image(image, preprocessing='caffe'):
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
        # RGB -> BGR
        image = image[:: -1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])
    elif preprocessing == 'torch':
        image /= 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    else:
        raise ValueError('Unknown preprocessing parameter.')
    return image


def extract_d2net_return(d2net, img_path, multi_scale=False, need_nms=False):
    image = imageio.imread(img_path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing='caffe',
    )
    with torch.no_grad():
        if multi_scale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=torch.device("cuda:0"),
                ),
                model=d2net,
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=torch.device("cuda:0"),
                ),
                d2net,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    return keypoints, descriptors, scores
