# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> aachen
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   04/08/2021 11:33
=================================================='''
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from tools.seg_tools import label_to_bgr, read_seg_map_without_group
from tools.common import sort_dict_by_value

from localization.coarse.coarselocalization import CoarseLocalization


def aachen():
    map_gid_rgb = read_seg_map_without_group("datasets/aachen/aachen_grgb_gid_v3.txt")
    # gid_rgb = {}
    # for k in map_gid_rgb.keys():
    #     gid = map_gid_rgb[k]
    #     b = k % 256
    #     r = k // (256 * 256)
    #     g = (k // 256) % 256
    #     gid_rgb[gid] = [r, g, b]
    save_root = "/data/cornucopia/fx221/exp/shloc/aachen"
    db_seg_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/global_seg_instance"
    # weight_name = "2021_08_03_11_06_14_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized"
    weight_name = "2021_08_05_23_29_59_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized"
    seg_dir = osp.join(save_root, weight_name, "vis")
    conf_dir = osp.join(save_root, weight_name, "confidence")
    # img_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright"
    q_imglist_full = "datasets/aachen/aachen_query_imglist.txt"
    q_imglist_fn = "datasets/aachen/fail_list.txt"

    # all_fns = []
    fail_list = []
    fail_fns = []
    with open(q_imglist_fn, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split('.')[0]
            fail_list.append(l)
    with open(q_imglist_full, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            img_fn = l.split('/')[-1].split('.')[0]
            if img_fn in fail_list:
                fail_fns.append(l)
    del fail_list

    with open("fail_imglist.txt", "w") as f:
        for fn in fail_fns:
            f.write(fn + "\n")
    exit(0)

    topk = 2
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("seg", cv2.WINDOW_NORMAL)
    for fn in fail_fns:
        print("fn: ", fn)
        img = cv2.imread(osp.join(seg_dir, fn.replace('jpg', 'png')))
        cv2.imshow("img", img)
        prediction = np.load(osp.join(conf_dir, fn.split('.')[0] + ".npy"), allow_pickle=True).item()
        conf = prediction["confidence"]
        labels = prediction["ids"]
        # segs = [cv2.resize(label_to_bgr(labels[i], maps=map_gid_rgb), dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        #         for i in range(topk)]
        confs = [conf[i] for i in range(topk)]

        for i in range(topk):
            label = labels[i]
            print("i: ", i, np.sum(label > 0))  # 2000

            mask_conf = (confs[i] < 1e-4)
            label[mask_conf] = 0
            # seg = label_to_bgr(label=label, maps=map_gid_rgb)
            uids = np.unique(label).tolist()
            valid_uids = [v for v in uids if v > 0]

            gid_conf = {}

            for v in valid_uids:
                # cnt = np.sum(label == v)
                cf = np.mean(confs[i][label == v])
                gid_conf[v] = cf
                # xy = np.where(label == v)
                # rgb = seg[xy[0][0], xy[1][0]]
                # bgr = [rgb[2], rgb[1], rgb[0]]
                # print("gid: bgr: conf: cnt: ", v, bgr, cf, cnt)

            sorted_gid_conf = sort_dict_by_value(data=gid_conf, reverse=True)
            new_label = np.zeros_like(label)
            nlbabels = 0
            for v, c in sorted_gid_conf:
                mask = (label == v)
                new_label[mask] = v
                # xy = np.where(label == v)
                # bgr = seg[xy[0][0], xy[1][0]]
                # rgb = [bgr[2], bgr[1], bgr[0]]
                rgb = map_gid_rgb[v]
                r = rgb // (256 * 256)
                g = (rgb // 256) % 256
                b = rgb % 256
                print("gid: conf: rgb", v, c, [r, g, b])
                nlbabels += 1
                if nlbabels >= 5:
                    break

            seg = label_to_bgr(label=new_label, maps=map_gid_rgb)
            cv2.namedWindow("seg{:d}".format(i), cv2.WINDOW_NORMAL)
            cv2.imshow("seg{:d}".format(i), seg)
            cv2.waitKey(0)


def get_coexit_pairs():
    db_seg_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/global_seg_instance"
    img_list_fn = '/home/mifs/fx221/Research/Code/shloc/datasets/aachen/aachen_db_query_imglist.txt'
    img_fns = []
    with open(img_list_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            img_fns.append(l)

    fn_gids = {}
    for fn in tqdm(img_fns, total=len(img_fns)):
        seg_fn = osp.join(db_seg_dir, fn.replace('jpg', 'png'))
        if not osp.exists(seg_fn):
            continue
        seg = cv2.imread(seg_fn)
        seg_lid = np.int32(seg[:, :, 2]) * 256 * 256 + np.int32(seg[:, :, 1]) * 256 + np.int32(seg[:, :, 0])
        uids = np.unique(seg_lid).tolist()
        valid_uids = [v for v in uids if v > 0]
        if len(valid_uids) == 0:
            continue
        fn_gids[fn] = valid_uids

    pairs = {}
    all_fns = sorted(fn_gids.keys())
    for fn in all_fns:
        pairs[fn] = []
    for i in tqdm(range(len(all_fns)), total=len(all_fns)):
        fn_i = all_fns[i]
        uids_i = fn_gids[fn_i]
        for j in range(i + 1, len(all_fns)):
            fn_j = all_fns[j]
            uids_j = fn_gids[fn_j]
            for id in uids_i:
                if id in uids_j:
                    if fn_i not in pairs[fn_j]:
                        pairs[fn_j].append(fn_i)
                    if fn_j not in pairs[fn_i]:
                        pairs[fn_i].append(fn_j)
                    break

    with open('pairs_coexist_global_instance.txt', 'w') as f:
        for fn in pairs.keys():
            cans = pairs[fn]
            for c in cans:
                f.write(fn + ' ' + c + '\n')


if __name__ == '__main__':
    # aachen()
    get_coexit_pairs()