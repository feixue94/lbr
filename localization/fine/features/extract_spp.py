# -*- coding: utf-8 -*-
"""
@Time ： 2021/2/8 下午1:28
@Auth ： Fei Xue
@File ： extract_spp.py
@Email： xuefei@sensetime.com
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
import os.path as osp
from tqdm import tqdm
from net.locnets.superpoint import SuperPointNet


def plot_keypoint(img_path, pts, scores=None):
    img = cv2.imread(img_path)
    img_out = img.copy()
    r = 5
    if scores is None:
        for pt in pts:
            img_out = cv2.circle(img_out, pt, r, (0, 0, 255), 4)
    else:
        scores_norm = scores / np.linalg.norm(scores, ord=2)
        print("score median: ", np.median(scores_norm))
        for i in range(pts.shape[0]):
            pt = pts[i]
            s = scores_norm[i]
            if s < np.median(scores_norm):
                continue
            # img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), int(r * s), (0, 0, 255), 4)
            img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), r, (0, 0, 255), 4)

    cv2.imshow("img", img_out)
    cv2.waitKey(0)

    # cv2.imwrite(img_path + ".d2net.png", img_out)


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    # rcorners = corners[:2, :].floor().astype(int)  # Rounded corners.
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    # print(np.max(rcorners[0, :]), np.max(corners[0, :]), H, W)
    # print(np.max(rcorners[1, :]), np.max(corners[1, :]), H, W)
    for i, rc in enumerate(rcorners.T):
        # print("i: ", i)
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def extract_spp(model, img, need_desc=False):
    semi, coarse_desc = model.det(img)
    H, W = img.size(2), img.size(3)
    semi = F.interpolate(semi.unsqueeze(1), size=[H, W], mode='bilinear')
    heatmap = semi.data.cpu().numpy().squeeze()
    # print("semi: ", semi[0, :, 0, 0])

    # Hc = semi.size(2)
    # Wc = semi.size(3)

    """
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi)  # Softmax.
    dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
    dense_max = np.max(dense, axis=0)
    dense_dustbin = dense[-1, :, :]

    print("median:", np.median(dense))
    print("diff:", dense_max - dense_dustbin)
    print("max diff: ", np.max(dense_max - dense_dustbin))
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.

    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, 8, 8])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc * 8, Wc * 8])

    """

    # dense = torch.exp(semi)
    # dense = dense / (torch.sum(dense, dim=1, keepdim=True) + .00001)
    # nodust = dense[:, :-1, :, :]
    # nodust = nodust.permute([0, 2, 3, 1])
    # heatmap = nodust.view((nodust.size(0), Hc, Wc, 8, 8))
    # heatmap = heatmap.permute([0, 1, 3, 2, 4])
    # heatmap = heatmap.contiguous().view((heatmap.size(0), Hc * 8, Wc * 8))
    # heatmap = heatmap.data.cpu().numpy().squeeze()

    # heatmap = np.exp(heatmap) / (np.sum(np.exp(heatmap)))
    # heatmap = heatmap / np.sum(heatmap)
    # print("heatmap: ", np.median(heatmap), np.max(heatmap))

    # conf_thresh = 0.5

    conf_thresh = 0.005
    # conf_thresh = 0.1
    # conf_thresh = 1e-3
    nms_dist = 4
    border_remove = 4
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]

    valid_idex = heatmap > conf_thresh
    valid_score = heatmap[valid_idex]

    if not need_desc:
        return pts, valid_score, None

    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy())
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.cuda()
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, heatmap, desc


def extract_spp_return(spp, img, need_nms=True, topK=-1, mask=None, **kwargs):
    with torch.no_grad():
        # img = cv2.imread(img_path, 0)
        # img = (img.astype('float32') / 255.)
        img -= torch.mean(img)
        # print(img.shape)
        # img = torch.from_numpy(img)
        # img = img.view(1, 1, img.size(0), img.size(1)).cuda()
        keypoints, scores, descriptors = extract_spp(model=spp, img=img.cuda(), need_desc=need_nms)

        keypoints = np.transpose(keypoints, [1, 0])
        descriptors = np.transpose(descriptors, [1, 0])
        scores = keypoints[:, 2]
        keypoints = keypoints[:, 0:2]

        # print(keypoints.shape, descriptors.shape, mask)

        if mask is not None:
            # cv2.imshow("mask", mask)
            # cv2.waitKey(0)
            labels = []
            others = []
            keypoints_with_labels = []
            scores_with_labels = []
            descriptors_with_labels = []
            keypoints_without_labels = []
            scores_without_labels = []
            descriptors_without_labels = []

            id_img = np.int32(mask[:, :, 2]) * 256 * 256 + np.int32(mask[:, :, 1]) * 256 + np.int32(mask[:, :, 0])
            # print(img.shape, id_img.shape)

            for i in range(keypoints.shape[0]):
                x = keypoints[i, 0]
                y = keypoints[i, 1]
                gid = id_img[int(y), int(x)]
                if gid == 0:
                    keypoints_without_labels.append(keypoints[i])
                    scores_without_labels.append(scores[i])
                    descriptors_without_labels.append(descriptors[i])
                    others.append(0)
                else:
                    keypoints_with_labels.append(keypoints[i])
                    scores_with_labels.append(scores[i])
                    descriptors_with_labels.append(descriptors[i])
                    labels.append(gid)

            # keypoints_with_labels = np.array(keypoints_with_labels)
            # scores_with_labels = np.array(scores_with_labels)
            # descriptors_with_labels = np.array(descriptors_with_labels)
            # labels = np.array(labels, np.int32)
            #
            # keypoints_without_labels = np.array(keypoints_without_labels)
            # scores_without_labels = np.array(scores_without_labels)
            # descriptors_without_labels = np.array(descriptors_without_labels)
            # others = np.array(others, np.int32)

            if topK > 0:
                if topK <= len(keypoints_with_labels):
                    idxes = np.array(scores_with_labels, np.float).argsort()[::-1][:topK]
                    keypoints = np.array(keypoints_with_labels, np.float)[idxes]
                    scores = np.array(scores_with_labels, np.float)[idxes]
                    labels = np.array(labels, np.int32)[idxes]
                    descriptors = np.array(descriptors_with_labels, np.float)[idxes]
                elif topK >= len(keypoints_with_labels) + len(keypoints_without_labels):
                    # keypoints = np.vstack([keypoints_with_labels, keypoints_without_labels])
                    # scores = np.vstack([scorescc_with_labels, scores_without_labels])
                    # descriptors = np.vstack([descriptors_with_labels, descriptors_without_labels])
                    # labels = np.vstack([labels, others])
                    keypoints = keypoints_with_labels
                    scores = scores_with_labels
                    descriptors = descriptors_with_labels
                    for i in range(len(others)):
                        keypoints.append(keypoints_without_labels[i])
                        scores.append(scores_without_labels[i])
                        descriptors.append(descriptors_without_labels[i])
                        labels.append(others[i])
                else:
                    n = topK - len(keypoints_with_labels)
                    idxes = np.array(scores_without_labels, np.float).argsort()[::-1][:n]
                    keypoints = keypoints_with_labels
                    scores = scores_with_labels
                    descriptors = descriptors_with_labels
                    for i in idxes:
                        keypoints.append(keypoints_without_labels[i])
                        scores.append(scores_without_labels[i])
                        descriptors.append(descriptors_without_labels[i])
                        labels.append(others[i])

            # print(keypoints.shape, descriptors.shape)
            return {"keypoints": np.array(keypoints, np.float),
                    "descriptors": np.array(descriptors, np.float),
                    "scores": np.array(scores, np.float),
                    "labels": np.array(labels, np.int32),
                    }
        else:
            # print(topK)
            if topK > 0:
                idxes = np.array(scores, np.float).argsort()[::-1][:topK]
                keypoints = np.array(keypoints[idxes], np.float)
                scores = np.array(scores[idxes], np.float)
                descriptors = np.array(descriptors[idxes], np.float)

            keypoints = np.array(keypoints, np.float)
            scores = np.array(scores, np.float)
            descriptors = np.array(descriptors, np.float)

            print(keypoints.shape, descriptors.shape)

            return {"keypoints": np.array(keypoints, np.float),
                    "descriptors": descriptors,
                    "scores": scores,
                    }
        # return {"keypoints": keypoints,
        #         "descriptors": descriptors,
        #         "scores": scores,
        #         }

        # return keypoints, descriptors, scores
