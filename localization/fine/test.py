# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> test
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   03/08/2021 12:12
=================================================='''
import os
import os.path as osp
import numpy as np
import torch
import cv2
from localization.fine.extractor import get_model
from localization.fine.matcher import Matcher
from localization.tools import plot_keypoint, plot_matches
from tools.common import resize_img

match_confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'NNM': {
        'output': 'NNM',
        'model': {
            'name': 'nnm',
            'do_mutual_check': True,
            'distance_threshold': None,
        },
    },
    'NNML': {
        'output': 'NNML',
        'model': {
            'name': 'nnml',
            'do_mutual_check': True,
            'distance_threshold': None,
        },
    },

    'ONN': {
        'output': 'ONN',
        'model': {
            'name': 'nn',
            'do_mutual_check': False,
            'distance_threshold': None,
        },
    },
    'NNR': {
        'output': 'NNR',
        'model': {
            'name': 'nnr',
            'do_mutual_check': True,
            'distance_threshold': 0.9,
        },
    }
}

extract_confs = {
    # 'superpoint_aachen': {
    'superpoint-n2000-r1024-mask': {
        'output': 'feats-superpoint-n2000-r1024-mask',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 2000,
            'model_fn': osp.join(os.getcwd(), "models/superpoint_v1.pth"),
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },

    'superpoint-n4096-r1024-mask': {
        'output': 'feats-superpoint-n4096-r1024-mask',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
            'model_fn': osp.join(os.getcwd(), "models/superpoint_v1.pth"),
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
        'mask': True,
    },

    'r2d2-n2000-r1024-mask': {
        'output': 'feats-r2d2-n2000-r1024-mask',
        'model': {
            'name': 'r2d2',
            'nms_radius': 4,
            'max_keypoints': 2000,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
        'mask': True,
    },

    'r2d2-n4096-r1024-mask': {
        'output': 'feats-r2d2-n2000-r1024-mask',
        'model': {
            'name': 'r2d2',
            'nms_radius': 4,
            'max_keypoints': 4096,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
        'mask': True,
    },
}


def test(use_mask=True):
    img_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright"
    mask_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/global_seg_instance"
    # pair_fn = "datasets/aachen/pairs-db-covis20.txt"
    # pair_fn = '/home/mifs/fx221/fx221/exp/shloc/aachen/2021_08_05_23_29_59_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized/loc_by_seg/loc_by_sec2_top30.txt'
    pair_fn = '/home/mifs/fx221/fx221/exp/shloc/aachen/2021_08_05_23_29_59_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized/loc_by_seg/loc_by_sec_top30_fail_list.txt'
    all_pairs = []
    with open(pair_fn, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(" ")
            all_pairs.append((l[0], l[1]))

    if use_mask:
        m_conf = match_confs["NNML"]
    else:
        m_conf = match_confs["NNM"]

    # e_conf = extract_confs["superpoint-n4096-r1024-mask"]
    e_conf = extract_confs["r2d2-n4096-r1024-mask"]
    matcher = Matcher(conf=m_conf)
    matcher = matcher.eval().cuda()

    model, extractor = get_model(model_name=e_conf['model']['name'], weight_path=e_conf["model"]["model_fn"])
    model = model.cuda()

    # matcher = Matcher[]

    cv2.namedWindow("pt0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pt1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)

    next_q = False
    for p in all_pairs:
        fn0 = p[0]
        fn1 = p[1]
        img0 = cv2.imread(osp.join(img_dir, fn0))
        img1 = cv2.imread(osp.join(img_dir, fn1))
        if e_conf["preprocessing"]["grayscale"]:
            img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)[None]
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)[None]
        else:
            img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img0_gray = img0_gray.transpose(2, 0, 1)
            img1_gray = img1_gray.transpose(2, 0, 1)
        mask0 = cv2.imread(osp.join(mask_dir, fn0.replace("jpg", "png")))
        mask0 = cv2.resize(mask0, dsize=(img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_NEAREST)
        # label0 = np.int32(mask0[:, :, 2]) * 256 * 256 + np.int32(mask0[:, :, 1]) * 256 + np.int32(mask0[:, :, 0])
        mask1 = cv2.imread(osp.join(mask_dir, fn1.replace("jpg", "png")))
        mask1 = cv2.resize(mask1, dsize=(img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
        # label1 = np.int32(mask1[:, :, 2]) * 256 * 256 + np.int32(mask1[:, :, 1]) * 256 + np.int32(mask1[:, :, 0])
        print(img0.shape, img1.shape)
        print(mask0.shape, mask1.shape)

        pred0 = extractor(model, img=torch.from_numpy(img0_gray / 255.).unsqueeze(0).float(),
                          topK=e_conf["model"]["max_keypoints"], mask=mask0 if use_mask else None)
        pred1 = extractor(model, img=torch.from_numpy(img1_gray / 255.).unsqueeze(0).float(),
                          topK=e_conf["model"]["max_keypoints"], mask=mask1 if use_mask else None)

        img_pt0 = plot_keypoint(img_path=img0, pts=pred0["keypoints"])
        img_pt1 = plot_keypoint(img_path=img1, pts=pred1["keypoints"])

        if use_mask:
            match_data = {
                "descriptors0": pred0["descriptors"],
                "labels0": pred0["labels"],
                "descriptors1": pred1["descriptors"],
                "labels1": pred1["labels"],
            }
        else:
            match_data = {
                "descriptors0": pred0["descriptors"],
                # "labels0": pred0["labels"],
                "descriptors1": pred1["descriptors"],
                # "labels1": pred1["labels"],
            }
        matches = matcher(match_data)["matches0"]
        # matches = pred['matches0']  # [0].cpu().short().numpy()
        valid_matches = []
        for i in range(matches.shape[0]):
            if matches[i] > 0:
                valid_matches.append([i, matches[i]])
        valid_matches = np.array(valid_matches, np.int)
        img_matches = plot_matches(img1=img0, img2=img1,
                                   pts1=pred0["keypoints"],
                                   pts2=pred1["keypoints"],
                                   matches=valid_matches,
                                   )

        img_pt0 = resize_img(img_pt0, nh=512)
        mask0 = resize_img(mask0, nh=512)
        img_pt1 = resize_img(img_pt1, nh=512)
        mask1 = resize_img(mask1, nh=512)
        img_matches = resize_img(img_matches, nh=512)

        cv2.imshow("pt0", np.hstack([img_pt0, mask0]))
        cv2.imshow("pt1", np.hstack([img_pt1, mask1]))
        cv2.imshow("match", img_matches)
        cv2.waitKey(0)


if __name__ == '__main__':
    test(use_mask=True)
