# -*- coding: utf-8 -*-
"""
@Time ： 2021/2/22 上午10:10
@Auth ： Fei Xue
@File ： extract_sift.py
@Email： fx221@cam.ac.uk
"""

import argparse
import os
import os.path as osp
from tqdm import tqdm
import cv2
import numpy as np


def plot_keypoint(img_path, pts, scores=None):
    img = cv2.imread(img_path)
    img_out = img.copy()
    r = 3
    if scores is None:
        for i in range(pts.shape[0]):
            pt = pts[i]
            # img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), int(r * s), (0, 0, 255), 4)
            img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), r, (0, 0, 255), 4)
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


def extract_sift_return(sift, img_path, need_nms=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=1, contrastThreshold=0.03, edgeThreshold=8)
    # kpts = sift.detect(img)
    # descs = np.zeros((10000, 128), np.float)
    kpts, descs = sift.detectAndCompute(img, None)

    scores = np.array([kp.response for kp in kpts], np.float32)
    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])

    return kpts, descs, scores


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='super point extraction on haptches')
    parser.add_argument('--hpatches_path', type=str, required=True,
                        help='path to a file containing a list of images to process')
    parser.add_argument("--output_path", type=str, required=True,
                        help='path to save descriptors')
    parser.add_argument("--image_list_file", type=str, default="img_list_hpatches_list.txt",
                        help='path to save descriptors')

    args = parser.parse_args()
    hpatches_path = args.hpatches_path
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    with open(args.image_list_file, 'r') as f:
        lines = f.readlines()

    sift = cv2.xfeatures2d.SIFT_create(4000)
    for line in tqdm(lines, total=len(lines)):
        path = line.strip()
        img_path = osp.join(args.hpatches_path, path)
        print("img_path: ", img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kpts = sift.detect(img)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        print(kpts.shape)
        plot_keypoint(img_path=img_path, pts=kpts)
