# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> resnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   07/09/2021 22:11
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder

import torchvision.transforms as tvf

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, groups=32, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, outplanes)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes, stride, groups, dilation)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = conv1x1(outplanes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNetX(nn.Module):
    def __init__(self, encoder_name='resnext101_32x4d', encoder_depth=3, encoder_weights='ssl', outdim=128,
                 freeze_encoder=False):
        super(ResNetX, self).__init__()

        encoder = get_encoder(name=encoder_name,
                              in_channels=3,
                              depth=encoder_depth,
                              weights=encoder_weights)

        if encoder_depth == 3:
            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,  # 2x ds
                encoder.relu,
                encoder.maxpool,
                encoder.layer1,  # 4x ds
                encoder.layer2,  # 8x ds
            )
            c = 512
            self.ds = 8

            self.conv = nn.Sequential(
                nn.Conv2d(c, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        elif encoder_depth == 2:
            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,  # 2x ds
                encoder.relu,
                encoder.maxpool,
                encoder.layer1,  # 4x ds
            )
            c = 256
            self.ds = 4

            self.conv = nn.Sequential(

                ResBlock(inplanes=256, outplanes=256, groups=32),
                ResBlock(inplanes=256, outplanes=256, groups=32),
                ResBlock(inplanes=256, outplanes=256, groups=32),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        elif encoder_depth == 1:
            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,  # 2x ds
                encoder.relu,
                encoder.maxpool,
            )
            c = 64
            self.ds = 2
            # TODO

        if freeze_encoder:
            print("Freeze encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Detector Head.
        self.convPa = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.convPb = torch.nn.Conv2d(256, self.ds * self.ds, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.convDb = torch.nn.Conv2d(256, outdim, kernel_size=1, stride=1, padding=0)

    def det(self, x):
        x = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(len(features))

        # pass through several CNN layers first
        x = self.conv(x)

        # Detector Head.
        cPa = self.convPa(x)
        semi = self.convPb(cPa)
        semi = torch.sigmoid(semi)

        Hc, Wc = semi.size(2), semi.size(3)
        # recover resolution
        semi = semi.permute([0, 2, 3, 1])
        score = semi.view(semi.size(0), Hc, Wc, self.ds, self.ds)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * self.ds, Wc * self.ds).unsqueeze(1)

        # Descriptor Head
        cDa = self.convDa(x)
        desc = self.convDb(cDa)
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
