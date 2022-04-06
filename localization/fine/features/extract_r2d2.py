# -*- coding: utf-8 -*-
"""
@Time ： 2021/2/8 下午1:28
@Auth ： Fei Xue
@File ： extract_spp.py
@Email： xuefei@sensetime.com
"""
import os, pdb
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.Normalize(mean=RGB_mean, std=RGB_std)])


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    # print("\n>> Creating net = " + checkpoint['net'])
    print("\n Loaded R2D2 model")
    net = eval(checkpoint['net'])
    # nb_of_weights = common.model_size(net)
    # print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")
    # print(" (Model size: {:.0f}K parameters )".format(nb_of_weights / 1000))

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale_org(net, img, detector, scale_f=2 ** 0.25,
                           min_scale=0.0, max_scale=1,
                           min_size=256, max_size=1024,
                           verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])

            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y, x = detector(**res)  # nms
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)
    return XYS.cpu().numpy(), D.cpu().numpy(), scores.cpu().numpy()


def extract_multiscale(net, img, detector, scale_f=2 ** 0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    pts_list = []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose:
                # print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
                print("extracting at scale x{:.02f} = {:4d}x{:3d}".format(s, nw, nh))
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])

            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            with torch.no_grad():
                y, x = detector(**res)  # nms
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)

            pts_list.append(torch.stack([x.float() * W / nw, y.float() * H / nh], dim=-1))

        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    # print("Y: ", len(Y))
    # print("X: ", len(X))
    # print("S: ", len(S))
    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)

    XYS = XYS.cpu().numpy()
    D = D.cpu().numpy()
    scores = scores.cpu().numpy()
    return XYS, D, scores


def extract(net, img, detector):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    with torch.no_grad():
        res = net(imgs=[img])

    # get output and reliability map
    descriptors = res['descriptors'][0]
    reliability = res['reliability'][0]
    repeatability = res['repeatability'][0]

    print("rel: ", torch.min(reliability), torch.max(reliability), torch.median(reliability))
    print("rep: ", torch.min(repeatability), torch.max(repeatability), torch.median(repeatability))

    # normalize the reliability for nms
    # extract maxima and descs
    with torch.no_grad():
        y, x = detector(**res)  # nms
    c = reliability[0, 0, y, x]
    q = repeatability[0, 0, y, x]
    d = descriptors[0, :, y, x].t()
    n = d.shape[0]

    print("after nms: ", n)

    X, Y, S, C, Q, D = [], [], [], [], [], []

    X.append(x.float())
    Y.append(y.float())
    C.append(c)
    Q.append(q)
    D.append(d)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    scores = torch.cat(C) * torch.cat(Q)

    XYS = torch.stack([Y, X], dim=-1)
    D = torch.cat(D)

    return XYS.cpu().numpy(), D.cpu().numpy(), scores.cpu().numpy()


def extract_r2d2_return(r2d2, img, need_nms=False, topK=-1, mask=None, **kwargs):
    # img = Image.open(img_path).convert('RGB')
    # H, W = img.size
    # print("r2d2_return: ", img.shape)
    img = norm_RGB(img.squeeze())
    img = img[None]
    img = img.cuda()

    rel_thr = 0.7  # 0.99
    rep_thr = 0.7  # 0.995
    min_size = 256
    max_size = 9999
    detector = NonMaxSuppression(rel_thr=rel_thr, rep_thr=rep_thr).cuda().eval()

    keypoints, descriptors, scores = extract_multiscale_org(net=r2d2, img=img, detector=detector,
                                                            min_size=min_size,
                                                            max_size=max_size, scale_f=1.2)  # r2d2 mode

    # print(keypoints.shape, descriptors.shape)

    keypoints = keypoints[:, 0:2]
    # pts = keypoints
    # pts[:, 1] = keypoints[:, 0]
    # pts[:, 0] = keypoints[:, 1]
    # keypoints = pts

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

    # return xys, desc, scores
