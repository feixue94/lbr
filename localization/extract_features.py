# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> extract_features
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   2021-05-13 10:13
=================================================='''

import torch
import numpy as np
import cv2
from net.locnets.superpoint import SuperPointNet, extract_spp_return
from net.locnets.r2d2 import extract_r2d2_return, Quad_L2Net_ConfCFS
from localization.tools import read_img

def extract_sift_return(sift, img, need_nms=False, **kwargs):
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kpts, descs = sift.detectAndCompute(img, None)

    scores = np.array([kp.response for kp in kpts], np.float32)
    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])

    return kpts, descs, scores

def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net'])
    net = eval(checkpoint['net'])
    # nb_of_weights = common.model_size(net)
    # print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")
    # print(" (Model size: {:.0f}K parameters )".format(nb_of_weights / 1000))

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()

def extract_feature(method, img_path, img_type=None, conf_th=0.015):
    color_type = 1
    if method == "spp":
        model = SuperPointNet()
        model.load_state_dict(torch.load('weights/superpoint_v1.pth'))
        model = model.cuda().eval()
        extractor = extract_spp_return
        color_type = 0
    elif method == "sift":
        model = cv2.xfeatures2d.SIFT_create(2000)
        extractor = extract_sift_return
        color_type = 0
    elif method == "r2d2":
        model_path = "weights/r2d2_WAF_N16.pt"
        model = load_network(model_fn=model_path).cuda().eval()
        extractor = extract_r2d2_return

    if not type(img_path) == list:
        img_lists = [img_path]
    else:
        img_lists = img_path

    all_pts = []
    all_descs = []
    all_scores = []
    for fn in img_lists:
        img = read_img(path=fn, color_type=color_type, type=img_type)
        pts, descs, scores = extractor(model, img=img, conf_thresh=conf_th, need_nms=True)

        all_pts.append(pts)
        all_descs.append(descs)
        all_scores.append(scores)

    return all_pts, all_descs, all_scores
