# -*- coding: utf-8 -*-
# @Time    : 2021/5/6 下午4:47
# @Author  : Fei Xue
# @Email   : fx221@cam.ac.uk
# @File    : tools.py
# @Software: PyCharm

import numpy as np
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
from localization.utm import from_latlon
import py360convert
import torch
from scipy.spatial.transform import Rotation as sciR


def compute_pose_error(pred_qcw, pred_tcw, gt_qcw, gt_tcw):
    pred_Rcw = sciR.from_quat(quat=[pred_qcw[1], pred_qcw[2], pred_qcw[3], pred_qcw[0]]).as_dcm()
    pred_tcw = np.array(pred_tcw, float).reshape(3, 1)
    pred_Rwc = pred_Rcw.transpose()
    pred_twc = -pred_Rcw.transpose() @ pred_tcw

    gt_Rcw = sciR.from_quat(quat=[gt_qcw[1], gt_qcw[2], gt_qcw[3], gt_qcw[0]]).as_dcm()
    gt_tcw = np.array(gt_tcw, float).reshape(3, 1)
    gt_Rwc = gt_Rcw.transpose()
    gt_twc = -gt_Rcw.transpose() @ gt_tcw

    t_error_xyz = pred_twc - gt_twc
    t_error = np.sqrt(np.sum(t_error_xyz ** 2))

    pred_qwc = sciR.from_quat(quat=[pred_qcw[1], pred_qcw[2], pred_qcw[3], pred_qcw[0]]).as_quat()
    gt_qwc = sciR.from_quat(quat=[gt_qcw[1], gt_qcw[2], gt_qcw[3], gt_qcw[0]]).as_quat()

    q_error = quaternion_angular_error(q1=pred_qwc, q2=gt_qwc)

    return q_error, t_error, (t_error_xyz[0, 0], t_error_xyz[1, 0], t_error_xyz[2, 0])


def assign_gps_by_interpolation(query_fns, base_gps):
    query_gps = {}
    for fn in query_fns:
        if fn in base_gps.keys():
            query_gps[fn] = base_gps[fn]
        else:
            left_id = -1
            right_id = -1
            left_gps = None
            right_gps = None
            tag = fn.split('.')[0]
            tag = tag.split('_')
            fn_cam, fn_id = tag[0], int(tag[-1])

            for k in base_gps.keys():
                tag_k = k.split('.')[0]
                tag_k = tag_k.split('_')
                k_cam, k_id = tag_k[0], int(tag_k[-1])
                k_gps = base_gps[k]
                if k_cam != fn_cam:
                    continue
                if k_id < fn_id:
                    if left_id == -1:
                        left_id = k_id
                        left_gps = k_gps
                    elif fn_id - k_id < fn_id - left_id:
                        left_id = k_id
                        left_gps = k_gps
                elif k_id > fn_id:
                    if right_id == -1:
                        right_id = k_id
                        right_gps = k_gps
                    elif k_id - fn_id < right_id - fn_id:
                        right_id = k_id
                        right_gps = k_gps

            if left_id == -1 and right_id == -1:
                continue
            elif left_id != -1 and right_id == -1:
                fn_gps = left_gps
            elif left_id == -1 and right_id != -1:
                fn_gps = right_gps
            else:
                g0 = left_gps[0] + (right_gps[0] - left_gps[0]) * (fn_id - left_id) / (right_id - left_id)
                g1 = left_gps[1] + (right_gps[1] - left_gps[1]) * (fn_id - left_id) / (right_id - left_id)
                fn_gps = (g0, g1)

            print("fn, left_fn, right_fn", fn_id, left_id, right_id)
            query_gps[fn] = fn_gps
    return query_gps


def plot_landmarks_from_gps(gps, center=None):
    """
    :param gps:[N, 2] [lat, lng]
    :return:
    """
    utms = np.zeros_like(gps)
    for i in range(gps.shape[0]):
        x, y, _, _ = from_latlon(latitude=gps[i, 0], longitude=gps[i, 1])
        utms[i, 0] = x
        utms[i, 1] = y

    print(np.mean(utms, axis=0))
    if center is not None:
        utms = utms - center
    else:
        utms = utms - np.mean(utms, axis=0)
    plt.figure(dpi=200)
    plt.plot(utms[:, 0], utms[:, 1], 'ob', markersize=1, )
    # markerfacecolor="red", markeredgewidth=6, markeredgecolor="grey")
    plt.savefig("gps.png")
    plt.show()


def plot_keypoint(img_path, pts, scores=None, tag=None, save_path=None):
    if type(img_path) == str:
        img = cv2.imread(img_path)
    else:
        img = img_path.copy()

    img_out = img.copy()
    print(img.shape)
    r = 3
    for i in range(pts.shape[0]):
        pt = pts[i]
        # img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), int(r * s), (0, 0, 255), 4)
        img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), r, (0, 0, 255), 2)

    if save_path is not None:
        cv2.imwrite(save_path, img_out)
    return img_out


def sort_dict_by_value(data, reverse=False):
    return sorted(data.items(), key=lambda d: d[1], reverse=reverse)


def plot_matches(img1, img2, pts1, pts2, matches, save_fn=None, horizon=True, window_name="img"):
    # img1 = cv2.imread(img_path1)
    # img2 = cv2.imread(img_path2)
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    r = 3
    if horizon:
        img_out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
        # Place the first image to the left
        img_out[:rows1, :cols1] = img1
        # Place the next image to the right of it
        img_out[:rows2, cols1:] = img2  # np.dstack([img2, img2, img2])
        for idx, pt in enumerate(pts1):
            img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), r, (0, 0, 255), 2)
        for idx, pt in enumerate(pts2):
            img_out = cv2.circle(img_out, (int(pt[0] + cols1), int(pt[1])), r, (0, 0, 255), 2)
        for idx, m in enumerate(matches):
            # if idx % 10 > 0:
            #     continue
            pt1 = pts1[m[0]]
            pt2 = pts2[m[1]]
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), r, (0, 255, 0), 2)

            img_out = cv2.circle(img_out, (int(pt2[0]) + cols1, int(pt2[1])), r, (0, 255, 0), 2)

            img_out = cv2.line(img_out, (int(pt1[0]), int(pt1[1])), (int(pt2[0]) + cols1, int(pt2[1])), (0, 255, 0),
                               2)
    else:
        img_out = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
        # Place the first image to the left
        img_out[:rows1, :cols1] = img1
        # Place the next image to the right of it
        img_out[rows1:, :cols2] = img2  # np.dstack([img2, img2, img2])
        for idx, pt in enumerate(pts1):
            img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), r, (0, 0, 255), 2)
        for idx, pt in enumerate(pts2):
            img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1] + rows1)), r, (0, 0, 255), 2)
        for idx, m in enumerate(matches):
            # if idx % 10 > 0:
            #     continue
            pt1 = pts1[m[0]]
            pt2 = pts2[m[1]]
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), r, (0, 255, 0), 2)

            img_out = cv2.circle(img_out, (int(pt2[0]), int(pt2[1]) + rows1), r, (0, 255, 0), 2)

            img_out = cv2.line(img_out, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]) + rows1), (0, 255, 0),
                               2)

    img_rs = cv2.resize(img_out, None, fx=0.5, fy=0.5)
    # img_rs = img_out  # cv2.resize(img_out, None, fx=0.5, fy=0.5)

    # if save_fn is not None:
    #     cv2.imwrite(save_fn, img_rs)
    # cv2.imshow(window_name, img_rs)
    # cv2.waitKey(10)
    return img_rs


def read_retrieval_results(path):
    output = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for p in lines:
            p = p.strip("\n").split(" ")

            if p[1] == "no_match":
                continue
            if p[0] in output.keys():
                output[p[0]].append(p[1])
            else:
                output[p[0]] = [p[1]]
    return output


def read_img(path, color_type=cv2.IMREAD_COLOR, type="cube"):
    img = cv2.imread(path, color_type)

    if type == "cube":
        if len(img.shape) == 2:
            img = img[:, :, None]
        face_w = img.shape[0] // 2
        img_cubes = py360convert.e2c(img, cube_format='dice', face_w=face_w, mode="bilinear")
        img = img_cubes[face_w:face_w * 2]
    return img


def nn_k(query_gps, db_gps, k=20):
    q = torch.from_numpy(query_gps)  # [N 2]
    db = torch.from_numpy(db_gps)  # [M, 2]
    # print (q.shape, db.shape)
    dist = q.unsqueeze(2) - db.t().unsqueeze(0)
    dist = dist[:, 0, :] ** 2 + dist[:, 1, :] ** 2
    print("dist: ", dist.shape)
    topk = torch.topk(dist, dim=1, k=k, largest=False)[1]
    return topk


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


if __name__ == '__main__':
    root = "/home/mifs/fx221/data/cam_street_view"
    query_img_path = "camvid_360_cvpr18_P2_training_data/images_hand"
    query_base_gps_file = "P2_training_gps.csv"
    query_base_gps = {}
    with open(osp.join(root, query_base_gps_file), "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip("\n").split(',')
            fn, lat, lng = l[0], float(l[1]), float(l[2])
            # query_base_gps[fn] = from_latlon(latitude=lat, longitude=lng)[0:2]
            query_base_gps[fn] = (lat, lng)
            # lat 52.196802, lng 0.130194

    query_img_lists = os.listdir(osp.join(root, query_img_path))
    query_gps = assign_gps_by_interpolation(query_fns=query_img_lists, base_gps=query_base_gps)

    with open("camvid_360_cvpr18_P2_training_data_prop_interpolation_gps.txt", "w") as f:
        for fn in sorted(query_gps.keys()):
            gps = query_gps[fn]
            text = "{:s} {:.10f} {:.10f}".format(fn, gps[0], gps[1])
            f.write(text + "\n")
