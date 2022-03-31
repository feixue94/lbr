# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> matchers
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   2021-05-13 10:48
=================================================='''

import torch
import numpy as np


# Mutual nearest neighbors matcher for L2 normalized descriptors.
def mutual_nn_matcher(descriptors1, descriptors2):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()


# Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def ratio_matcher(descriptors1, descriptors2, ratio=0.9):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    # print("sim: ", sim.shape)

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]

    # Symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)

    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


# Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.9):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]

    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))

    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


def matcher_with_label(pts1, descs1, pts2, descs2, labels1, labels2, global_ids=None, matcher="nn"):
    def sel_pts_with_labels(pts, descs, labels):
        sel_pts = []
        sel_descs = []
        nums = [0]
        n = 0
        for gid in global_ids:
            if gid == 0:
                continue
            for idx, p in enumerate(pts):
                id = labels[int(p[1]), int(p[0])]
                if gid == id:
                    sel_pts.append(p)
                    sel_descs.append(descs[idx])
                    n += 1
            nums.append(n)
        return np.array(sel_pts, pts.dtype), np.array(sel_descs, descs.dtype), nums

    if global_ids is None:
        global_ids = np.unique(labels1).tolist()

    sel_pts1, sel_descs1, nums1 = sel_pts_with_labels(pts=pts1, descs=descs1, labels=labels1)
    sel_pts2, sel_descs2, nums2 = sel_pts_with_labels(pts=pts2, descs=descs2, labels=labels2)

    # print(nums1)
    # print(nums2)

    matches = []
    for n in range(1, len(nums1)):
        s1, e1 = nums1[n - 1], nums1[n]
        s2, e2 = nums2[n - 1], nums2[n]
        if s1 == e1 or s2 == e2:
            continue

        if s1 >= e1 - 2 or s2 >= e2 - 2:
            continue
        # print(s1, e1)
        # print(s2, e2)
        if matcher == "nn":
            n_matches = mutual_nn_matcher(descriptors1=torch.FloatTensor(sel_descs1[s1:e1]).cuda(),
                                          descriptors2=torch.FloatTensor(sel_descs2[s2:e2]).cuda())
        elif matcher == "ratio":
            n_matches = ratio_matcher(descriptors1=torch.FloatTensor(sel_descs1[s1:e1]).cuda(),
                                      descriptors2=torch.FloatTensor(sel_descs2[s2:e2]).cuda(),
                                      ratio=0.9)
        elif matcher == "nn_ratio":
            n_matches = mutual_nn_ratio_matcher(descriptors1=torch.FloatTensor(sel_descs1[s1:e1]).cuda(),
                                                descriptors2=torch.FloatTensor(sel_descs2[s2:e2]).cuda(),
                                                ratio=0.9)
        for m in n_matches:
            matches.append([m[0] + s1, m[1] + s2])

    matches = np.array(matches, np.int)
    return sel_pts1, sel_descs1, sel_pts2, sel_descs2, matches


def matcher_corss_label(pts1, descs1, pts2, descs2, labels1, labels2, global_ids=None):
    def sel_pts_with_labels(pts, descs, labels):
        sel_pts = []
        sel_descs = []
        nums = [0]
        n = 0
        for gid in global_ids:
            if gid == 0:
                continue
            for idx, p in enumerate(pts):
                id = labels[int(p[1]), int(p[0])]
                if gid == id:
                    sel_pts.append(p)
                    sel_descs.append(descs[idx])
                    n += 1
            nums.append(n)
        return np.array(sel_pts, pts.dtype), np.array(sel_descs, descs.dtype), nums

    if global_ids is None:
        global_ids = np.unique(labels1).tolist()

    sel_pts1, sel_descs1, nums1 = sel_pts_with_labels(pts=pts1, descs=descs1, labels=labels1)
    sel_pts2, sel_descs2, nums2 = sel_pts_with_labels(pts=pts2, descs=descs2, labels=labels2)

    # print(nums1)
    # print(nums2)

    matches = []
    for n in range(1, len(nums1)):
        s1, e1 = nums1[n - 1], nums1[n]
        s2, e2 = nums2[n - 1], nums2[n]
        if s1 == e1 or s2 == e2:
            continue
        # n_matches = mutual_nn_matcher(descriptors1=torch.FloatTensor(sel_descs1[s1:e1]).cuda(),
        #                               descriptors2=torch.FloatTensor(sel_descs2[s2:e2]).cuda())
        n_matches = ratio_matcher(descriptors1=torch.FloatTensor(sel_descs1[s1:e1]).cuda(),
                                  descriptors2=torch.FloatTensor(sel_descs2[s2:e2]).cuda())
        for m in n_matches:
            matches.append([m[0] + s1, m[1] + s2])

    matches = np.array(matches, np.int)
    return sel_pts1, sel_descs1, sel_pts2, sel_descs2, matches
