# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> resnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/09/2021 22:11
=================================================='''

import numpy as np
import torch.nn.functional as F
import torch

import time


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def det(self, x):
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        semi = torch.exp(semi)
        semi = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        semi = semi[:, :-1, :, :]
        Hc, Wc = semi.size(2), semi.size(3)
        semi = semi.permute([0, 2, 3, 1])
        score = semi.view(semi.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)
        # score = torch.softmax()

        # return score
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return score, desc

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        semi = torch.exp(semi)
        semi = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        semi = semi[:, :-1, :, :]
        Hc, Wc = semi.size(2), semi.size(3)
        semi = semi.permute([0, 2, 3, 1])
        score = semi.view(semi.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return score, desc


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


def extract_spp(model, img, conf_thresh=0.015, need_desc=False):
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

    # print("conf_th: ", conf_thresh)
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


def extract_spp_return(spp, img, need_nms=True, conf_thresh=0.005, topK=-1):
    with torch.no_grad():
        # img = cv2.imread(img_path, 0)
        img = (img.astype('float32') / 255.)
        img -= np.mean(img)
        img = torch.from_numpy(img)
        img = img.view(1, 1, img.size(0), img.size(1)).cuda()
        keypoints, scores, descriptors = extract_spp(model=spp, img=img, need_desc=need_nms, conf_thresh=conf_thresh)

        keypoints = np.transpose(keypoints, [1, 0])
        descriptors = np.transpose(descriptors, [1, 0])
        scores = keypoints[:, 2]

        return keypoints[:, 0:2], descriptors, scores


if __name__ == '__main__':
    img = torch.ones((1, 1, 480, 640)).cuda()
    # net = CSPD2Net32(outdim=128, use_bn=True).cuda()

    total_time = 0
    # net = SPD2NetSPP(outdim=128, use_bn=True).cuda()
    net = SuperPointNet().cuda()

    for i in range(1000):
        start_time = time.time()
        out = net(img)

        total_time = total_time + time.time() - start_time

    print("mean time: ", total_time / 1000)
