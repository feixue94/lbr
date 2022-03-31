# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> spd2
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   20/08/2021 17:51
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as tvf

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.Normalize(mean=RGB_mean, std=RGB_std)])


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


class SPD2L2Net(nn.Module):
    def __init__(self, outdim=128, freeze_encoder=False):
        super(SPD2L2Net, self).__init__()
        self.outdim = outdim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),

        )

        if freeze_encoder:
            print("Freeze the encoder")
            for param in self.conv1.parameters():
                param.requires_grad = False

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        self.convPb = nn.Conv2d(128, 1, kernel_size=1, padding=0)

    def det(self, x):
        x = self.conv1(x)
        score = self.convPb(x)
        score = torch.sigmoid(score)
        desc = self.convDb(x)
        desc = F.normalize(desc, dim=1)

        return score, desc

    def forward(self, batch):
        b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        score, desc = self.det(x)

        return {
            "score": score,
            "desc": desc
        }

        desc1 = desc[:b, :, :, :]
        desc2 = desc[b:, :, :, :]

        score1 = score[:b, :, :]
        score2 = score[b:, :, :]

        return {
            'dense_features1': desc1,
            'scores1': score1,
            'dense_features2': desc2,
            'scores2': score2,
        }


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


def extrat_spd2_return(model, img, conf_th=0.001, scale_f=2 ** 0.25,
                       min_scale=0.05, max_scale=1.0,
                       min_size=256, max_size=2048,
                       mask=None, topK=-1, **kwargs):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # print("img: ", img.shape)
    # exit(0)

    img = norm_RGB(img.squeeze())
    img = img[None]
    img = img.cuda()

    B, one, H, W = img.shape

    assert max_scale <= 1
    s = 1.0

    # print(img.shape)

    # X, Y, S, C, Q, D = [], [], [], [], [], []
    all_pts, all_scores, all_descs = [], [], []
    all_pts_list = []

    # print(min_size / max(H, W))
    # print(max_size / max(H, W))

    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        # print("hhh")
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]

            with torch.no_grad():
                heatmap, coarse_desc = model.det(img)

                # print("nh, nw, heatmap: ", nh, nw, heatmap.shape, coarse_desc.shape)

            if len(heatmap.size()) == 3:
                heatmap = heatmap.unsqueeze(1)
            if len(heatmap.size()) == 2:
                heatmap = heatmap.unsqueeze(0)
                heatmap = heatmap.unsqueeze(1)
            # print(heatmap.shape)
            if heatmap.size(2) != nh or heatmap.size(3) != nw:
                heatmap = F.interpolate(heatmap, size=[nh, nw], mode='bilinear', align_corners=False)

            heatmap = heatmap.data.cpu().numpy().squeeze()

            conf_thresh = conf_th
            nms_dist = 4
            border_remove = 4
            xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.

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

            # """
            # --- Process descriptor.
            # coarse_desc = coarse_desc.data.cpu().numpy().squeeze()
            D = coarse_desc.size(1)
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                if coarse_desc.size(2) == nh and coarse_desc.size(3) == nw:
                    desc = coarse_desc[:, :, pts[1, :], pts[0, :]]
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                else:
                    # Interpolate into descriptor map using 2D point locations.
                    samp_pts = torch.from_numpy(pts[:2, :].copy())
                    samp_pts[0, :] = (samp_pts[0, :] / (float(nw) / 2.)) - 1.
                    samp_pts[1, :] = (samp_pts[1, :] / (float(nh) / 2.)) - 1.
                    samp_pts = samp_pts.transpose(0, 1).contiguous()
                    samp_pts = samp_pts.view(1, 1, -1, 2)
                    samp_pts = samp_pts.float()
                    samp_pts = samp_pts.cuda()
                    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

            pts[0, :] = pts[0, :] * W / nw
            pts[1, :] = pts[1, :] * H / nh

            pts = np.transpose(pts, [1, 0])
            all_pts.append(pts)
            all_scores.append(pts[:, 2])
            all_descs.append(np.transpose(desc, [1, 0]))

            all_pts_list.append(pts)

            # print("heatmap: ", heatmap.shape, pts.shape)
            # print(valid_score.shape)
            # print(desc.shape)

        s /= scale_f
        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    torch.backends.cudnn.benchmark = old_bm

    ns = len(all_pts)
    if ns == 0:
        return None, None, None
    keypoints = np.vstack(all_pts)
    scores = keypoints[:, 2]
    keypoints = keypoints[:, 0:2]
    descriptors = np.vstack(all_descs)
    # print("extract {:d} features from {:d} scales".format(all_pts.shape[1], ns))

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
            # print("x-y", x, y, int(x), int(y))
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
        keypoints = np.array(keypoints, np.float)
        descriptors = np.array(descriptors, np.float)
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

        # print(keypoints.shape, descriptors.shape)

        return {"keypoints": np.array(keypoints, np.float),
                "descriptors": descriptors,
                "scores": scores,
                }
    # return all_pts, all_descs, all_scores


def extract_resnet_return(model, img, conf_th=0.001,
                          mask=None,
                          topK=-1,
                          **kwargs):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    img = norm_RGB(img.squeeze())
    img = img[None]
    img = img.cuda()

    B, one, H, W = img.shape

    all_pts = []
    all_descs = []

    if 'scales' in kwargs.keys():
        scales = kwargs.get('scales')
    else:
        scales = [1.0]

    for s in scales:
        if s == 1.0:
            new_img = img
        else:
            nh = int(H * s)
            nw = int(W * s)
            new_img = F.interpolate(img, size=(nh, nw), mode='bilinear')
        nh, nw = new_img.shape[2:]

        with torch.no_grad():
            heatmap, coarse_desc = model.det(new_img)

            # print("nh, nw, heatmap, desc: ", nh, nw, heatmap.shape, coarse_desc.shape)
            if len(heatmap.size()) == 3:
                heatmap = heatmap.unsqueeze(1)
            if len(heatmap.size()) == 2:
                heatmap = heatmap.unsqueeze(0)
                heatmap = heatmap.unsqueeze(1)
            # print(heatmap.shape)
            if heatmap.size(2) != nh or heatmap.size(3) != nw:
                heatmap = F.interpolate(heatmap, size=[nh, nw], mode='bilinear', align_corners=False)


            conf_thresh = conf_th
            nms_dist = 8
            border_remove = 8
            # heatmap = heatmap.data.cpu().numpy().squeeze()
            # xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
            #
            # pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
            # pts[0, :] = ys
            # pts[1, :] = xs
            # pts[2, :] = heatmap[xs, ys]
            #
            # pts, _ = nms_fast(pts, nh, nw, dist_thresh=nms_dist)  # Apply NMS.

            scores = simple_nms(heatmap, nms_radius=3)
            keypoints = [
                torch.nonzero(s > conf_thresh)
                for s in scores]
            scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
            # print(keypoints[0].shape)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]
            scores = scores[0].data.cpu().numpy().squeeze()
            keypoints = keypoints[0].data.cpu().numpy().squeeze()
            pts = keypoints.transpose()
            pts[2, :] = scores
            # print(pts.shape, keypoints.shape)
            # pts = np.zeros((3, keypoints.shape[0]))  # Populate point data sized 3xN.
            # print(len(keypoints), keypoints[0].shape, keypoints[0][0])
            # exit(0)
            # pts[0, :] = keypoints[:, 1]
            # pts[1, :] = keypoints[:, 0]
            # pts[2, :] = scores

            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Sort by confidence.
            # Remove points along border.
            bord = border_remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]

            # valid_idex = heatmap > conf_thresh
            # valid_score = heatmap[valid_idex]
            # """
            # --- Process descriptor.
            # coarse_desc = coarse_desc.data.cpu().numpy().squeeze()
            D = coarse_desc.size(1)
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                if coarse_desc.size(2) == nh and coarse_desc.size(3) == nw:
                    desc = coarse_desc[:, :, pts[1, :], pts[0, :]]
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                else:
                    # Interpolate into descriptor map using 2D point locations.
                    samp_pts = torch.from_numpy(pts[:2, :].copy())
                    samp_pts[0, :] = (samp_pts[0, :] / (float(nw) / 2.)) - 1.
                    samp_pts[1, :] = (samp_pts[1, :] / (float(nh) / 2.)) - 1.
                    samp_pts = samp_pts.transpose(0, 1).contiguous()
                    samp_pts = samp_pts.view(1, 1, -1, 2)
                    samp_pts = samp_pts.float()
                    samp_pts = samp_pts.cuda()
                    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

            if pts.shape[1] == 0:
                continue

            pts[0, :] = pts[0, :] * W / nw
            pts[1, :] = pts[1, :] * H / nh
            all_pts.append(np.transpose(pts, [1, 0]))
            all_descs.append(np.transpose(desc, [1, 0]))

    all_pts = np.vstack(all_pts)
    all_descs = np.vstack(all_descs)

    torch.backends.cudnn.benchmark = old_bm

    if all_pts.shape[0] == 0:
        return None, None, None

    # keypoints = np.transpose(pts, [1, 0])
    # descriptors = np.transpose(desc, [1, 0])
    # scores = keypoints[:, 2]
    # keypoints = keypoints[:, 0:2]

    keypoints = all_pts[:, 0:2]
    scores = all_pts[:, 2]
    descriptors = all_descs

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
            # print("x-y", x, y, int(x), int(y))
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
        keypoints = np.array(keypoints, np.float)
        descriptors = np.array(descriptors, np.float)
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

        # print(keypoints.shape, descriptors.shape)

        return {"keypoints": np.array(keypoints, np.float),
                "descriptors": descriptors,
                "scores": scores,
                }
    # return all_pts, all_descs, all_scores
