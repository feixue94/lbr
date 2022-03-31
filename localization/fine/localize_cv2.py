import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py
from tqdm import tqdm
import pickle
import pycolmap
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import math
import os.path as osp
from scipy.spatial.transform import Rotation as sciR
from tools.common import sort_dict_by_value

from localization.utils.read_write_model import read_model
from localization.utils.parsers import (
    parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair)

from localization.fine.matcher import Matcher, confs
from localization.p3d.projection import reproject, reproject_fromR, calc_depth, ColmapQ2R, compute_pose_error
# from localization.p3d.projection import draw_camera
from localization.p3d.ploter import plot_matches, plot_reprojpoint2D
from tools.common import resize_img


# def calc_uncertainty(mkp, mp3d, qvec, tvec, camera, r_samples=[0], t_samples=[0]):
#     cmap = plt.get_cmap('jet')
#     R = sciR.from_quat(quat=[qvec[1], qvec[2], qvec[3], qvec[0]]).as_dcm()
# 
#     H = camera['height']
#     W = camera['width']
# 
#     results = {}
#     n_sample = 0
#     all_probs = []
#     for rx in r_samples:
#         for ry in r_samples:
#             for rz in r_samples:
#                 for tx in t_samples:
#                     for ty in t_samples:
#                         for tz in t_samples:
#                             dR = sciR.from_euler('zyx', [rz, ry, rx], degrees=True).as_dcm()
#                             # print(dR)
#                             # print(dR.shape)
#                             dt = np.array([tx, ty, tz], np.float).reshape(3, 1)
#                             nR = R @ dR
#                             nt = tvec.reshape(3, 1) + dt
#                             proj_kp = reproject_fromR(points3D=mp3d, rot=nR, tvec=nt, camera=camera)
#                             mask = (proj_kp[:, 0] >= 0) * (proj_kp[:, 0] <= W) * (proj_kp[:, 1] >= 0) * (
#                                     proj_kp[:, 1] < H)
#                             proj_error = (proj_kp - mkp) ** 2
#                             proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1]) / 20.0
#                             # sum_proj = np.sum(proj_error[mask])
#                             # prob = np.exp(-sum_proj)
#                             prob = np.exp(-np.median(proj_error))
# 
#                             results[n_sample] = {
#                                 'rt': (rx, ry, rz, tx, ty, tz),
#                                 'prob': prob,
#                                 'proj_error': proj_error,
#                                 'Pcw': np.hstack([dR.reshape(3, 3), dt.reshape(3, 1)]),
#                                 'rgb': cmap(prob),
#                             }
#                             n_sample += 1
# 
#                             print('prob: ', prob)
#                             all_probs.append(prob)
#     print('all_porbs: ', np.min(all_probs), np.median(all_probs), np.max(all_probs))
#     # exit(0)
# 
#     # visualize camera poses
#     pangolin.CreateWindowAndBind('Main', 640, 480)
#     gl.glEnable(gl.GL_DEPTH_TEST)
#     gl.glEnable(gl.GL_BLEND)
#     gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
# 
#     # Define Projection and initial ModelView matrix
#     scam = pangolin.OpenGlRenderState(
#         pangolin.ProjectionMatrix(1024, 1024, 2000, 2000, 512, 512, 0.1, 1000),
#         pangolin.ModelViewLookAt(0, -100, -0.1, 0, 0, 0, 0.0, -1.0, 0.0))
#     handler = pangolin.Handler3D(scam)
# 
#     # Create Interactive View in window
#     dcam = pangolin.CreateDisplay()
#     dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
#     dcam.SetHandler(handler)
# 
#     # Twc = pangolin.OpenGlMatrix.SetIdentity()
#     while not pangolin.ShouldQuit():
#         gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
#         gl.glClearColor(1.0, 1.0, 1.0, 1.0)
#         dcam.Activate(scam)
# 
#         # Render OpenGL Cube
#         # pangolin.glDrawColouredCube()
# 
#         # Draw Point Cloud
#         # points = np.random.random((100000, 3)) * 10
#         # gl.glPointSize(2)
#         # gl.glColor3f(1.0, 0.0, 0.0)
#         # pangolin.DrawPoints(points)
# 
#         for v in results.keys():
#             Pcw = results[v]['Pcw']
#             twc = - (Pcw[0:3, 0:3]).transpose() @ Pcw[0:3, -1]
#             Rwc = (Pcw[:3, 0:3]).transpose()
#             Pwc = np.eye(4, dtype=np.float64)
#             Pwc[0:3, 0:3] = Rwc
#             Pwc[0:3, -1] = twc
#             rgb = results[v]['rgb']
#             # twc = twc.astype(np.float64)
#             # print('twc: ', twc, type(twc))
#             draw_camera(Twc=Pwc.transpose(), color=(rgb[0], rgb[1], rgb[2]), camera_size=0.01, linewidth=2)
# 
#         pangolin.FinishFrame()

# def quaternion_angular_error(q1, q2):
#     """
#     angular error between two quaternions
#     :param q1: (4, )
#     :param q2: (4, )
#     :return:
#     """
#     d = abs(np.dot(q1, q2))
#     d = min(1.0, max(-1.0, d))
#     theta = 2 * np.arccos(d) * 180 / np.pi
#     return theta


def is_cluster(H, W, points2D, radius=50, ratio=0.6):
    assert points2D.shape[0] > 0
    n_points = np.zeros(shape=(H, W), dtype=np.int)
    for i in range(points2D.shape[0]):
        x = int(points2D[i, 0])
        y = int(points2D[i, 0])

        for ry in range(-radius, radius):
            for rx in range(-radius, radius):
                if rx < 0 or rx >= W or ry < 0 or ry >= H:
                    continue

                n_points[y + ry, x + rx] += 1

    n_most = np.max(n_points)

    return (n_most / points2D.shape[0] >= ratio)


def calc_dist(qname, feature_file, db_images, points3D, qid_p3ds, obs_th=0):
    desc_q = feature_file[qname]['descriptors'].__array__()
    desc_q = desc_q.transpose()

    all_median_dist = []
    for qid in qid_p3ds.keys():
        p3d = qid_p3ds[qid]
        if p3d == -1:
            continue

        if len(points3D[p3d].image_ids) < obs_th:
            continue

        q_desc_i = desc_q[qid]
        db_ids = points3D[p3d].image_ids
        all_dist = []
        for db_id in db_ids:
            db_3Ds = db_images[db_id].point3D_ids
            db_name = db_images[db_id].name
            desc_db = feature_file[db_name]['descriptors'].__array__()
            desc_db = desc_db.transpose()
            db_pki = list(db_3Ds).index(p3d)
            db_desc_i = desc_db[db_pki]
            dist = q_desc_i @ db_desc_i.reshape(-1, 1)
            dist = np.sqrt(2 - 2 * dist + 1e-6)

            all_dist.append(dist)

        if len(all_dist) == 0:
            continue

        all_median_dist.append(np.median(all_dist))

    return np.median(all_median_dist)


def do_covisibility_clustering(frame_ids, all_images, points3D):
    clusters = []
    visited = set()

    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = all_images[exploration_frame].point3D_ids
            connected_frames = set(
                j for i in observed if i != -1 for j in points3D[i].image_ids)
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def get_covisibility_frames(frame_id, all_images, points3D, covisibility_frame=50, ref_3Dpoints=None, obs_th=0):
    # visited = set()
    # cluster = []
    # queue = {frame_id}
    # while len(queue):
    #     exploration_frame = queue.pop()
    #
    #     if exploration_frame in visited:
    #         continue
    #
    #     visited.add(exploration_frame)
    #     cluster.append(exploration_frame)
    #     observed = all_images[exploration_frame].point3D_ids
    #     connected_frames = set(
    #         j for i in observed if i != -1 for j in points3D[i].image_ids)
    #     # connected_frames &= set([frame_id])
    #     connected_frames -= visited
    #     queue |= connected_frames

    if ref_3Dpoints is not None:
        observed = ref_3Dpoints
        connected_frames = [j for i in ref_3Dpoints if i != -1 for j in points3D[i].image_ids]
        connected_frames = np.unique(connected_frames)
    else:
        observed = all_images[frame_id].point3D_ids
        connected_frames = [j for i in observed if i != -1 for j in points3D[i].image_ids]
        connected_frames = np.unique(connected_frames)
    print('Find {:d} connected frames'.format(len(connected_frames)))
    valid_db_ids = []
    db_obs = {}
    for db_id in connected_frames:
        p3d_ids = all_images[db_id].point3D_ids
        covisible_p3ds = [v for v in observed if v != -1 and len(points3D[v].image_ids) >= obs_th and v in p3d_ids]
        db_obs[db_id] = len(covisible_p3ds)

    sorted_db_obs = sort_dict_by_value(data=db_obs, reverse=True)
    for item in sorted_db_obs:
        valid_db_ids.append(item[0])

        if covisibility_frame > 0:
            if len(valid_db_ids) >= covisibility_frame:
                break

    # if frame_id not in valid_db_ids:
    #     valid_db_ids.append(frame_id)

    print('Retain {:d} valid connected frames'.format(len(valid_db_ids)))
    return valid_db_ids


def pose_refinement_covisibility(qname, cfg, feature_file, db_frame_id, db_images, points3D, thresh, matcher,
                                 with_label=False,
                                 covisibility_frame=50,
                                 ref_3Dpoints=None,
                                 plus05=False,
                                 iters=1,
                                 obs_th=3,
                                 opt_th=12,
                                 qvec=None,
                                 tvec=None,
                                 radius=20,
                                 log_info='',
                                 opt_type="ref",
                                 image_dir=None,
                                 vis_dir=None,
                                 depth_th=0,
                                 gt_qvec=None,
                                 gt_tvec=None,
                                 ):
    db_ids = get_covisibility_frames(frame_id=db_frame_id, all_images=db_images, points3D=points3D,
                                     covisibility_frame=covisibility_frame, ref_3Dpoints=ref_3Dpoints,
                                     obs_th=obs_th)

    kpq = feature_file[qname]['keypoints'].__array__()
    desc_q = feature_file[qname]['descriptors'].__array__()
    score_q = feature_file[qname]['scores'].__array__()
    desc_q = desc_q.transpose()
    if with_label:
        label_q = feature_file[qname]['labels'].__array__()
    else:
        label_q = None

    # do matching between query and candidate frames
    mp3d = []
    mkpq = []
    mkpdb = []
    # all_obs = []
    all_3D_ids = []
    all_score_q = []
    qid_p3ds = {}
    valid_qid_mp3d_ids = {}
    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        kpdb = feature_file[db_name]['keypoints'].__array__()

        desc_db = feature_file[db_name]["descriptors"].__array__()
        desc_db = desc_db.transpose()

        points3D_ids = db_images[db_id].point3D_ids
        if points3D_ids.size == 0:
            print("No 3D points in this db image: ", db_name)
            continue

        if with_label:
            label_db = feature_file[db_name]["labels"].__array__()
            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=label_q, label_db=label_db,
                                       matcher=matcher,
                                       db_3D_ids=points3D_ids)
        else:
            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=None, label_db=None,
                                       matcher=matcher,
                                       db_3D_ids=points3D_ids)

        # matches = matcher(match_data)["matches0"]

        valid = np.where(matches > -1)[0]
        valid = valid[points3D_ids[matches[valid]] != -1]

        # matched_mkq = []
        # matched_mkdb = []
        inliers = []

        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            if len(points3D[id_3D].image_ids) < obs_th:
                continue

            if idx in valid_qid_mp3d_ids.keys():
                if id_3D in valid_qid_mp3d_ids[idx]:
                    continue
                else:
                    valid_qid_mp3d_ids[idx].append(id_3D)
            else:
                valid_qid_mp3d_ids[idx] = [id_3D]

            if idx in qid_p3ds.keys():
                if id_3D in qid_p3ds[idx]:
                    continue
                else:
                    qid_p3ds[idx].append(id_3D)
            else:
                qid_p3ds[idx] = [id_3D]

            # matched_mkq.append(kpq[idx])
            # matched_mkdb.append(kpdb[matches[idx]])

            if qvec is not None and tvec is not None:
                proj_2d = reproject(points3D=np.array(points3D[id_3D].xyz).reshape(-1, 3), rvec=qvec, tvec=tvec,
                                    camera=cfg)

                proj_error = (kpq[idx] - proj_2d) ** 2
                proj_error = np.sqrt(np.sum(proj_error))
                if proj_error > radius:
                    inliers.append(False)
                    continue

            inliers.append(True)

            mp3d.append(points3D[id_3D].xyz)
            mkpq.append(kpq[idx])
            mkpdb.append(kpdb[matches[idx]])
            all_3D_ids.append(id_3D)

            all_score_q.append(score_q[idx])

            # all_obs.append(len(points3D[id_3D].image_ids))

        ### visualize matches
        '''
        q_img = cv2.imread(osp.join(image_dir, qname))
        db_img = cv2.imread(osp.join(image_dir, db_name))
        inliers = np.array(inliers, np.uint8)
        matched_mkq = np.array(matched_mkq).reshape(-1, 2)
        matched_mkdb = np.array(matched_mkdb).reshape(-1, 2)
        img_match = plot_matches(img1=q_img, img2=db_img, pts1=matched_mkq, pts2=matched_mkdb, inliers=inliers, plot_outlier=True)
        img_match = resize_img(img_match, nh=512)
        cv2.imshow("match-q-db", img_match)
        cv2.waitKey(5)
        
        if vis_dir is not None:
            save_fn = 'exmatch_with_proj_{:s}_{:s}.png'.format(qname.replace('/', '-'), db_name.replace('/', '-'))
            cv2.imwrite(osp.join(vis_dir, save_fn), img_match)
        '''

    mp3d = np.array(mp3d, float).reshape(-1, 3)
    mkpq = np.array(mkpq, float).reshape(-1, 2)

    mkpq = mkpq + 0.5

    print_text = 'Get {:d} covisible frames with {:d} matches from cluster optimization'.format(len(db_ids),
                                                                                                mp3d.shape[0])
    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')

    ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, opt_th)
    init_qvec = ret['qvec']
    init_tvec = ret['tvec']
    proj_mkp = reproject(mp3d, rvec=qvec, tvec=tvec, camera=cfg)
    proj_error = (mkpq - proj_mkp) ** 2
    proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
    inlier_mask = (np.array(ret['inliers'], int) > 0)
    mn_error = np.min(proj_error[inlier_mask])
    md_error = np.median(proj_error[inlier_mask])
    mx_error = np.max(proj_error[inlier_mask])

    depth = calc_depth(points3D=mp3d, rvec=qvec, tvec=tvec, camera=cfg)
    mn_depth = np.min(depth[inlier_mask])
    md_depth = np.median(depth[inlier_mask])
    mx_depth = np.max(depth[inlier_mask])

    # q_diff = quaternion_angular_error(q1=np.array(qvec, np.float).reshape(4, ),
    #                                   q2=np.array(init_qvec, np.float).reshape(4, ))
    # t_diff = (np.array(tvec, np.float).reshape(3, 1) - np.array(init_tvec, np.float).reshape(3, 1)) ** 2
    # t_diff = np.sqrt(np.sum(t_diff))
    # q_diff, t_diff, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=init_qvec, gt_tcw=init_tvec)
    q_diff, t_diff, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=gt_qvec, gt_tcw=gt_tvec)
    print_text = 'Iter: {:d} inliers: {:d} mn_error: {:.2f}, md_error: {:.2f} mx_error: {:.2f} mn_d:{:.1f} md_d:{:.1f} mx_d:{:.1f}, q_error:{:.1f} t_error:{:.2f}'.format(
        0,
        ret['num_inliers'],
        mn_error,
        md_error,
        mx_error,
        mn_depth,
        md_depth,
        mx_depth,
        q_diff,
        t_diff,
    )

    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')

    inliers_rsac = ret['inliers']
    # init_qvec = ret['qvec']
    # init_tvec = ret['tvec']
    if opt_type.find("ref") >= 0:
        for i in range(iters):
            inlier_mask_opt = []
            for pi in range(proj_error.shape[0]):
                if proj_error[pi] <= opt_th and inliers_rsac[pi]:
                    # if proj_error[pi] <= opt_th:
                    keep = True
                else:
                    keep = False
                # if np.sum(inlier_mask) > 100:
                #     if depth_th > 0:
                #         if depth[pi] > depth_th or depth[pi] <= 0:
                #             keep = False
                #     elif depth_th == -1.0:
                #         if depth[pi] > md_depth or depth[pi] <= 0:
                #             keep = False
                inlier_mask_opt.append(keep)

            ret = pycolmap.pose_refinement(tvec, qvec, mkpq, mp3d, inlier_mask_opt, cfg)
            # mkpq = mkpq[inlier_mask_opt]
            # mp3d = mp3d[inlier_mask_opt]
            # all_3D_ids = np.array(all_3D_ids)[inlier_mask_opt]
            # all_score_q = np.array(all_score_q)[inlier_mask_opt]
            # ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, opt_th)
            #
            qvec = ret['qvec']
            tvec = ret['tvec']

            proj_mkp = reproject(mp3d, rvec=qvec, tvec=tvec, camera=cfg)
            proj_error = (mkpq - proj_mkp) ** 2
            proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])

            depth = calc_depth(points3D=mp3d, rvec=qvec, tvec=tvec, camera=cfg)
            # inlier_mask = np.array(inlier_mask_opt).reshape(-1,)
            inlier_mask = (proj_error <= opt_th)  # np.array(inlier_mask_opt).reshape(-1,)

            # if depth_th > 0:
            #     inlier_mask = inlier_mask * (depth <= depth_th) * (depth > 0)

            # print('heiheihei - inliers: ', proj_error[inlier_mask].shape)
            mn_error = np.min(proj_error[inlier_mask])
            md_error = np.median(proj_error[inlier_mask])
            mx_error = np.max(proj_error[inlier_mask])

            mn_depth = np.min(depth[inlier_mask])
            md_depth = np.median(depth[inlier_mask])
            mx_depth = np.max(depth[inlier_mask])

            q_diff, t_diff, _ = compute_pose_error(pred_qcw=ret['qvec'], pred_tcw=ret['tvec'], gt_qcw=gt_qvec,
                                                   gt_tcw=gt_tvec)
            print_text = 'After Iter:{:d} inliers:{:d}/{:d} mn_error:{:.1f}, md_error:{:.1f} mx_error:{:.1f} mn_d:{:.1f} md_d:{:.1f} mx_d:{:.1f}, q_error:{:.1f}, t_error:{:.2f}'.format(
                i + 1,
                np.sum(
                    inlier_mask),
                np.sum(inlier_mask_opt),
                mn_error,
                md_error,
                mx_error,
                mn_depth,
                md_depth,
                mx_depth,
                q_diff,
                t_diff
            )
            print(print_text)
            if log_info is not None:
                log_info += (print_text + "\n")

            ret['inliers'] = inlier_mask_opt
            ret['num_inliers'] = np.sum(inlier_mask_opt)

    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')

    ret['mkpq'] = mkpq
    ret['3D_ids'] = all_3D_ids
    ret['db_ids'] = db_ids
    ret['score_q'] = all_score_q
    ret['log_info'] = log_info
    return ret


def pose_refinment_covisibility_by_projection(qname, cfg, feature_file, db_frame_id, db_images, points3D, thresh,
                                              matcher,
                                              qvec,
                                              tvec,
                                              q_p3d_ids,
                                              radius=10,
                                              n_can_3Ds=50,
                                              with_label=False,
                                              covisibility_frame=50,
                                              ref_3Dpoints=None,
                                              plus05=False,
                                              iters=1,
                                              log_info=None,
                                              opt_type="ref",
                                              obs_th=0,
                                              opt_th=12,
                                              image_dir=None,
                                              with_dist=False,
                                              ):
    db_ids = get_covisibility_frames(frame_id=db_frame_id, all_images=db_images, points3D=points3D,
                                     covisibility_frame=covisibility_frame, ref_3Dpoints=ref_3Dpoints, obs_th=obs_th)

    kpq = feature_file[qname]['keypoints'].__array__()
    desc_q = feature_file[qname]['descriptors'].__array__()
    desc_q = desc_q.transpose()
    if with_label:
        label_q = feature_file[qname]['labels'].__array__()
    else:
        label_q = None

    n_kpq = kpq.shape[0]
    H = cfg['height']
    W = cfg['width']
    rs = [i for i in range(-radius, radius + 1)]
    # print('rs: ', rs)
    pos_kp_ids = {}
    qid_3Did = {}
    qid_score = {}
    for i in range(n_kpq):
        u = kpq[i, 0]
        v = kpq[i, 1]
        for dv in rs:
            for du in rs:
                nv = int(v + dv)
                nu = int(u + du)
                if nu < 0 or nu >= W or nv < 0 or nv >= H:
                    continue
                id = nv * W + nu

                if id in pos_kp_ids.keys():
                    pos_kp_ids[id].append(i)
                else:
                    pos_kp_ids[id] = [i]

    if with_dist:
        dist_thresh = calc_dist(qname=qname, db_images=db_images, points3D=points3D, qid_p3ds=q_p3d_ids,
                                feature_file=feature_file, obs_th=obs_th)
    else:
        dist_thresh = 10
    print("dist_th: ", dist_thresh)

    for i, db_id in enumerate(db_ids):
        points3D_ids = db_images[db_id].point3D_ids
        db_name = db_images[db_id].name
        if points3D_ids.size == 0:
            print('No 3D points in db image: ', db_name)
            continue

        #### test reprojection
        '''
        mkdbi = feature_file[db_name]['keypoints'].__array__()
        valid_mkdb = []
        valid_mp3d = []
        obs = []
        for pid, p3d in enumerate(points3D_ids):
            if p3d == -1:
                continue

            valid_mkdb.append(mkdbi[pid])
            valid_mp3d.append(points3D[p3d].xyz)
            obs.append(int(len(points3D[p3d].image_ids)))

        obs = np.array(obs, np.int)
        valid_mkdb = np.array(valid_mkdb, np.float).reshape(-1, 2)
        valid_mp3d = np.array(valid_mp3d, np.float).reshape(-1, 3)
        proj_valid_mkdb = reproject(points3D=valid_mp3d, rvec=qvec, tvec=tvec, camera=cfg)

        q_img = cv2.imread(osp.join(image_dir, qname))
        db_img = cv2.imread(osp.join(image_dir, db_name))
        inliers = [True for k in range(proj_valid_mkdb.shape[0]) if len(points3D)]
        inliers = np.array(inliers, np.uint8)
        mask = (proj_valid_mkdb[:, 0] >= 0) * (proj_valid_mkdb[:, 0] < H) * (proj_valid_mkdb[:, 1] >= 0 ) * (proj_valid_mkdb[:, 1] <H) * (obs > 5)
        valid_mkdb = valid_mkdb[mask]
        proj_valid_mkdb = proj_valid_mkdb[mask]
        inliers = inliers[mask]

        img_match = plot_matches(img1=q_img, img2=db_img, pts1=proj_valid_mkdb, pts2=valid_mkdb, inliers=inliers)
        img_match = resize_img(img_match, nh=512)
        cv2.imshow("img_match_proj", img_match)
        cv2.waitKey(0)
        '''

        desc_db = feature_file[db_name]["descriptors"].__array__()
        desc_db = desc_db.transpose()
        for db_kpi, id_3D in enumerate(points3D_ids):
            if id_3D == -1:
                continue

            # if len(points3D[id_3D].image_ids) < obs_th:
            #     continue

            xyz = points3D[id_3D].xyz
            proj_uv = reproject(xyz.reshape(1, 3), rvec=qvec, tvec=tvec, camera=cfg)
            # print('proj_uv: ', proj_uv, proj_uv.shape)
            pu = proj_uv[0, 0]
            pv = proj_uv[0, 1]
            if pu < 0 or pu >= W or pv < 0 or pv >= H:
                continue

            db_desc_i = desc_db[db_kpi]

            puv = int(pv) * W + int(pu)
            if puv not in pos_kp_ids.keys():
                continue
            cans = pos_kp_ids[puv]

            best_score = -1
            best_q_id = -1
            for q_id in cans:
                q_desc_i = desc_q[q_id]
                score = q_desc_i @ db_desc_i.reshape(-1, 1)
                dist = np.sqrt(2 - 2 * score + 1e-6)

                # apply dist for rejection
                if dist > dist_thresh:
                    continue

                if score > best_score:
                    best_score = score
                    best_q_id = q_id

            if best_q_id == -1:
                continue

            if best_q_id not in qid_3Did.keys():
                qid_3Did[best_q_id] = id_3D
                qid_score[best_q_id] = best_score
            else:
                if best_score > qid_score[best_q_id]:
                    qid_score[best_q_id] = best_score
                    qid_3Did[best_q_id] = id_3D

            # for c in cans:
            #     if c not in qid_3Did.keys():
            #         qid_3Did[c] = [id_3D]
            #     else:
            #         qid_3Did[c].append(id_3D)
    mp3d = []
    mkpq = []
    all_3D_ids = []
    discard_qid_id_3D = {}
    matched_qid_3D = {}

    for q_id in q_p3d_ids.keys():
        q_id_3Ds = q_p3d_ids[q_id]
        for id_3D in [q_id_3Ds]:
            if len(points3D[id_3D].image_ids) < obs_th:
                discard_qid_id_3D[q_id] = id_3D
                continue

            if q_id in matched_qid_3D.keys():
                if id_3D in matched_qid_3D[q_id]:
                    continue
                else:
                    matched_qid_3D[q_id].append(id_3D)
            else:
                matched_qid_3D[q_id] = [id_3D]

            mp3d.append(points3D[id_3D].xyz)
            mkpq.append(kpq[q_id])
            all_3D_ids.append(id_3D)

    for q_id in qid_3Did.keys():
        q_id_3Ds = qid_3Did[q_id]
        for id_3D in [q_id_3Ds]:
            if len(points3D[id_3D].image_ids) < obs_th:
                discard_qid_id_3D[q_id] = id_3D
                continue

            if q_id in matched_qid_3D.keys():
                if id_3D in matched_qid_3D[q_id]:
                    continue
                else:
                    matched_qid_3D[q_id].append(id_3D)
            else:
                matched_qid_3D[q_id] = [id_3D]

            mp3d.append(points3D[id_3D].xyz)
            mkpq.append(kpq[q_id])
            all_3D_ids.append(id_3D)

    # if len(mp3d) < 100:
    #     for q_id in discard_qid_id_3D.keys():
    #         id_3D = discard_qid_id_3D[q_id]
    #
    #         if len(points3D[id_3D].image_ids) < obs_th / 2:
    #             discard_qid_id_3D[q_id] = id_3D
    #             continue
    #         if q_id in qid_3Did.keys():
    #             if id_3D == qid_3Did[q_id]:
    #                 continue
    #         mp3d.append(points3D[id_3D].xyz)
    #         mkpq.append(kpq[q_id])
    #         all_3D_ids.append(id_3D)

    if len(mp3d) < n_can_3Ds:
        print('Find insufficient {:d} 3d points by reprojection'.format(len(mp3d)))
        return {'success': False, 'num_inliers': 0, 'qvec': None, 'tvec': None}

    print('Find {:d} matched 3D points by reprojection'.format(len(mp3d)))

    mp3d = np.array(mp3d, np.float).reshape(-1, 3)
    mkpq = np.array(mkpq, np.float).reshape(-1, 2)

    if plus05:
        mkpq = mkpq + 0.5

    if opt_type.find("ras") >= 0:
        # ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)
        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, opt_th)
        # print(ret['inliers'])
        inlier_mask_opt = ret['inliers']
        qvec = ret['qvec']
        tvec = ret['tvec']
        proj_mkp = reproject(mp3d, rvec=qvec, tvec=tvec, camera=cfg)
        proj_error = (mkpq - proj_mkp) ** 2
        proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
        inlier_mask = np.array(inlier_mask_opt).reshape(-1, )
        mn_error = np.min(proj_error[inlier_mask])
        md_error = np.median(proj_error[inlier_mask])
        mx_error = np.max(proj_error[inlier_mask])
        print_text = 'Find {:d}/{:d} inliers after RANSAC with mn_error: {:.2f}, md_error: {:.2f} mx_error: {:.2f}'.format(
            ret['num_inliers'],
            np.sum(inlier_mask),
            mn_error,
            md_error,
            mx_error)
        print(print_text)
        if log_info is not None:
            log_info += (print_text + "\n")

    # """
    # inlier_mask = np.array([True for i in range(mkpq.shape[0])], np.uint8).reshape(-1, 1)
    # inlier_mask = (mkpq[:, 0] >= 0)  # all matches are inliers
    if opt_type.find("ref") >= 0:
        for i in range(iters):
            inlier_mask_opt = []
            for pi in range(proj_error.shape[0]):
                if proj_error[pi] <= opt_th:
                    inlier_mask_opt.append(True)
                else:
                    inlier_mask_opt.append(False)

            ret = pycolmap.pose_refinement(tvec, qvec, mkpq, mp3d, inlier_mask_opt, cfg)

            proj_mkp = reproject(mp3d, rvec=qvec, tvec=tvec, camera=cfg)
            proj_error = (mkpq - proj_mkp) ** 2
            proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
            # inlier_mask = np.array(inlier_mask_opt).reshape(-1,)
            inlier_mask = (proj_error <= opt_th)  # np.array(inlier_mask_opt).reshape(-1,)
            # print('heiheihei - inliers: ', proj_error[inlier_mask].shape)
            mn_error = np.min(proj_error[inlier_mask])
            md_error = np.median(proj_error[inlier_mask])
            mx_error = np.max(proj_error[inlier_mask])
            print_text = 'After Iter: {:d} inliers: {:d}/{:d} mn_error: {:.2f}, md_error: {:.2f} mx_error: {:.2f}'.format(
                i + 1,
                np.sum(
                    inlier_mask),
                np.sum(inlier_mask_opt),
                mn_error,
                md_error,
                mx_error)
            print(print_text)
            if log_info is not None:
                log_info += (print_text + "\n")

    ret['num_inliers'] = np.sum(inlier_mask)
    ret['mkpq'] = mkpq
    ret['3D_ids'] = all_3D_ids
    ret['inliers'] = inlier_mask_opt
    ret['db_ids'] = db_ids
    # """
    # for logging
    inlier_3D_ids = []
    for i in range(len(all_3D_ids)):
        if inlier_mask[i]:
            inlier_3D_ids.append(all_3D_ids[i])

    for i, db_id in enumerate(db_ids):
        points3D_ids = db_images[db_id].point3D_ids
        db_name = db_images[db_id].name
        if points3D_ids.size == 0:
            print('No 3D points in db image: ', db_name)
            continue

        matched_3D_ids = [v for v in inlier_3D_ids if v in points3D_ids]
        print_text = "Find {:d} inliers with {:s}".format(len(matched_3D_ids), db_name)
        print(print_text)
        if log_info is not None:
            log_info += (print_text + "\n")

    print_text = "Find {:d}/{:d} unique 3D points".format(len(inlier_3D_ids), len(
        [v for v in inlier_3D_ids if inlier_3D_ids.count(v) == 1]))
    print(print_text)
    if log_info is not None:
        log_info += (print_text + "\n")
    return {**ret, **{'log_info': log_info}}


def pose_from_cluster(qname, qinfo, db_ids, db_images, points3D,
                      feature_file, match_file, thresh):
    kpq = feature_file[qname]['keypoints'].__array__()
    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0

    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        points3D_ids = db_images[db_id].point3D_ids

        pair = names_to_pair(qname, db_name)
        matches = match_file[pair]['matches0'].__array__()
        valid = np.where(matches > -1)[0]
        # print ("valid: ", valid)
        # print ("matches: ", matches)
        # print ("qname: ", qname, db_name)
        # exit(0)
        if points3D_ids.size == 0:
            continue  # handles case where image does not see 3D points
        valid = valid[points3D_ids[matches[valid]] != -1]
        num_matches += len(valid)

        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mkpq = kpq[mkp_idxs]
    mkpq += 0.5  # COLMAP coordinates

    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    mp3d = [points3D[j].xyz for j in mp3d_ids]
    mp3d = np.array(mp3d).reshape(-1, 3)

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]

    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }
    ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)
    ret['cfg'] = cfg
    return ret, mkpq, mp3d, mp3d_ids, num_matches, (mkp_idxs, mkp_to_3D_to_db)


def pose_from_single(qname, qinfo, db_ids, db_images, points3D,
                     feature_file, match_file, thresh, image_dir):
    print("qname: ", qname)
    q_img = cv2.imread(osp.join(image_dir, qname))
    kpq = feature_file[qname]['keypoints'].__array__()

    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }

    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        print("db_name: ", db_name)

        db_img = cv2.imread(osp.join(image_dir, db_name))

        kpdb = feature_file[db_name]['keypoints'].__array__()

        points3D_ids = db_images[db_id].point3D_ids

        pair = names_to_pair(qname, db_name)
        matches = match_file[pair]['matches0'].__array__()
        valid = np.where(matches > -1)[0]

        if points3D_ids.size == 0:
            print("No 3D points in this db image: ", db_name)
            continue

        valid = valid[points3D_ids[matches[valid]] != -1]

        mp3d = []
        mkpq = []
        mkpdb = []
        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            mp3d.append(points3D[id_3D].xyz)

            mkpq.append(kpq[idx])
            mkpdb.append(kpdb[matches[idx]])

            # p3d_idxes.append(points3D_ids[matches[idx]])

        mp3d = np.array(mp3d, np.float).reshape(-1, 3)
        mkpq = np.array(mkpq, np.float).reshape(-1, 2) + 0.5

        if mp3d.shape[0] < 10:
            print("qnqme: {:s} dbname: {:s} failed because of insufficient matches {:d}".format(qname, db_name,
                                                                                                mp3d.shape[0]))
            continue

        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)

        if not ret['success'] and ret['num_inliers'] >= 30:
            print("qname: {:s} dbname: {:s} failed after optimization".format(qname, db_name))
            continue

        # localization succeed
        qvec = ret['qvec']
        tvec = ret['tvec']
        ret['cfg'] = cfg
        inliers = ret['inliers']
        num_inliers = ret['num_inliers']
        matches = np.array(ret['inliers'], np.bool).reshape(-1, 1)

        img_match = plot_matches(img1=q_img, img2=db_img,
                                 pts1=mkpq, pts2=mkpdb,
                                 inliers=matches)

        cv2.namedWindow("match", cv2.WINDOW_NORMAL)
        cv2.imshow("match", img_match)
        cv2.waitKey(5)

        return qvec, tvec, inliers, num_inliers

    # print("Try to localization {:s} failed".format(qname))
    closest = db_images[db_ids[0]]
    return closest.qvec, closest.tvec, [], 0


def feature_matching(desc_q, desc_db, matcher, label_q=None, label_db=None, db_3D_ids=None):
    with_label = (label_q is not None and label_db is not None)
    # print(desc_q.shape, desc_db.shape, db_3D_ids.shape)
    if db_3D_ids is None:
        if with_label:
            match_data = {
                "descriptors0": desc_q,
                "labels0": label_q,
                "descriptors1": desc_db,
                "labels1": label_db,
            }
        else:
            match_data = {
                "descriptors0": desc_q,
                "descriptors1": desc_db,
            }

        # keep the order: 1st: query, 2nd: db
        matches = matcher(match_data)["matches0"]
        return matches
    else:  # perform matching between desc_q and desc_db (with valid 3D points)
        masks = (db_3D_ids != -1)
        # valid_ids = np.where(db_3D_ids != -1)
        valid_desc_db = desc_db[masks]
        valid_ids = [i for i in range(desc_db.shape[0]) if masks[i]]

        if np.sum(masks) <= 3:
            return np.ones((desc_q.shape[0],), np.int) * -1

        if with_label:
            valid_label_db = label_db[masks]
            match_data = {
                "descriptors0": desc_q,
                "labels0": label_q,
                "descriptors1": valid_desc_db,
                "labels1": valid_label_db,
            }
        else:
            match_data = {
                "descriptors0": desc_q,
                "descriptors1": valid_desc_db,
            }

        # keep the order: 1st: query, 2nd: db
        matches = matcher(match_data)["matches0"]
        # print('matches: ', matches.shape)
        for i in range(desc_q.shape[0]):
            if matches[i] >= 0:
                matches[i] = valid_ids[matches[i]]
    return matches


def match_cluster_2D(kpq, desc_q, label_q, db_ids, points3D, feature_file, db_images, with_label, matcher,
                     plus05=False, obs_th=0):
    all_mp3d = []
    all_mkpq = []
    all_mp3d_ids = []
    all_q_ids = []
    outputs = {}

    valid_2D_3D_matches = {}
    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        kpdb = feature_file[db_name]['keypoints'].__array__()
        desc_db = feature_file[db_name]["descriptors"].__array__()
        desc_db = desc_db.transpose()

        points3D_ids = db_images[db_id].point3D_ids
        if points3D_ids.size == 0:
            print("No 3D points in this db image: ", db_name)
            continue

        # print("desc_q/desc_db", desc_q.shape, desc_db.shape, db_name)
        if with_label:
            label_db = feature_file[db_name]["labels"].__array__()

            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=label_q, label_db=label_db,
                                       matcher=matcher,
                                       db_3D_ids=points3D_ids,
                                       )
        else:
            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=None, label_db=None,
                                       matcher=matcher, db_3D_ids=points3D_ids)

        mkpdb = []
        mp3d_ids = []
        q_ids = []
        mkpq = []
        mp3d = []
        valid_matches = []
        for idx in range(matches.shape[0]):
            if matches[idx] == -1:
                continue
            if points3D_ids[matches[idx]] == -1:
                continue
            id_3D = points3D_ids[matches[idx]]

            # reject 3d points without enough observations
            if len(points3D[id_3D].image_ids) < obs_th:
                continue

            # remove duplicated matches
            if idx in valid_2D_3D_matches.keys():
                if id_3D in valid_2D_3D_matches[idx]:
                    continue
                else:
                    valid_2D_3D_matches[idx].append(id_3D)
            else:
                valid_2D_3D_matches[idx] = [id_3D]

            mp3d.append(points3D[id_3D].xyz)
            mp3d_ids.append(id_3D)
            all_mp3d_ids.append(id_3D)

            mkpq.append(kpq[idx])
            mkpdb.append(kpdb[matches[idx]])
            q_ids.append(idx)
            all_q_ids.append(idx)

            all_mkpq.append(kpq[idx])
            all_mp3d.append(points3D[id_3D].xyz)

            valid_matches.append(matches[idx])

        outputs[db_name] = {}
        outputs[db_name]['mkpq'] = mkpq
        outputs[db_name]['mkpdb'] = mkpdb
        outputs[db_name]['qids'] = q_ids
        outputs[db_name]['matches'] = np.array(valid_matches, np.int)
        outputs[db_name]['mp_3d_ids'] = mp3d_ids
        outputs[db_name]['mp3d'] = np.array(mp3d, np.float).reshape(-1, 3)

        print('Find {:d} valid matches from {:d}th candidate'.format(len(valid_matches), i))

    all_mp3d = np.array(all_mp3d, float).reshape(-1, 3)
    all_mkpq = np.array(all_mkpq, float).reshape(-1, 2)

    all_mkpq = all_mkpq + 0.5

    return outputs, all_mp3d, all_mkpq, all_mp3d_ids, all_q_ids


def pose_from_cluster_with_matcher(qname, qinfo, db_ids, db_images, points3D,
                                   feature_file,
                                   thresh,
                                   image_dir,
                                   matcher,
                                   do_covisility_opt=False,
                                   with_label=False,
                                   vis_dir=None,
                                   inlier_th=10,
                                   covisibility_frame=50,
                                   global_score=None,
                                   seg_dir=None,
                                   q_seg=None,
                                   log_info=None,
                                   opt_type="cluster",
                                   plus05=False,
                                   do_cluster_check=False,
                                   iters=1,
                                   radius=0,
                                   obs_th=0,
                                   opt_th=12,
                                   with_dist=False,
                                   inlier_ths=None,
                                   retrieval_sources=None,
                                   depth_th=0,
                                   gt_qvec=None,
                                   gt_tvec=None,
                                   ):
    print("qname: ", qname)
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    q_img = cv2.imread(osp.join(image_dir, qname))
    kpq = feature_file[qname]['keypoints'].__array__()
    scoreq = feature_file[qname]['scores'].__array__()
    desc_q = feature_file[qname]['descriptors'].__array__()
    desc_q = desc_q.transpose()

    if with_label:
        label_q = feature_file[qname]['labels'].__array__()
    else:
        label_q = None

    # results = {}
    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }

    best_results = {
        'tvec': None,
        'qvec': None,
        'num_inliers': 0,
        'single_num_inliers': 0,
        'db_id': -1,
        'order': -1,
        'qname': qname,
        'optimize': False,
        'dbname': db_images[db_ids[0][0]].name,
        "ret_source": "",
        "inliers": [],
    }

    for cluster_idx, db_id_cls in enumerate(db_ids):
        if inlier_ths is not None:
            inlier_th = inlier_ths[cluster_idx]
        if retrieval_sources is not None:
            ret_source = retrieval_sources[cluster_idx]
        else:
            ret_source = ""

        db_id = db_id_cls[0]
        db_name = db_images[db_id].name
        cluster_info, mp3d, mkpq, mp3d_ids, q_ids = match_cluster_2D(kpq=kpq, desc_q=desc_q, label_q=label_q,
                                                                     db_ids=db_id_cls,
                                                                     points3D=points3D,
                                                                     feature_file=feature_file,
                                                                     db_images=db_images,
                                                                     with_label=with_label,
                                                                     matcher=matcher,
                                                                     plus05=plus05,
                                                                     obs_th=3,
                                                                     )

        if mp3d.shape[0] < inlier_th:
            print_text = "qname: {:s} dbname: {:s}({:d}/{:d}) failed because of insufficient 3d points {:d}".format(
                qname,
                db_name,
                cluster_idx + 1,
                len(
                    db_ids),
                mp3d.shape[
                    0])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)

        if not ret["success"]:
            print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed after optimization".format(qname, db_name,
                                                                                                 cluster_idx + 1,
                                                                                                 len(db_ids))
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        inliers = ret['inliers']
        inlier_p3d_ids = [mp3d_ids[i] for i in range(len(inliers)) if inliers[i]]

        # visualize the matches
        q_p3d_ids = np.zeros(shape=(desc_q.shape[0], 1), dtype=np.int) - 1
        for idx, qid in enumerate(q_ids):
            if inliers[idx]:
                q_p3d_ids[qid] = mp3d_ids[idx]

        best_dbname = None
        best_inliers = -1
        for db_name in cluster_info.keys():
            matched_mp3d_ids = cluster_info[db_name]['mp_3d_ids']
            matched_qids = cluster_info[db_name]['qids']
            n = 0
            for idx, qid in enumerate(matched_qids):
                if matched_mp3d_ids[idx] == q_p3d_ids[qid]:
                    n += 1
            if n > best_inliers:
                best_inliers = n
                best_dbname = db_name

        if best_dbname is not None:
            # print('best_dbname: ', best_dbname)
            vis_matches = cluster_info[best_dbname]['matches']
            vis_p3d_ids = cluster_info[best_dbname]['mp_3d_ids']
            vis_mkpdb = cluster_info[best_dbname]['mkpdb']
            vis_mkpq = cluster_info[best_dbname]['mkpq']
            vis_mp3d = cluster_info[best_dbname]['mp3d']
            vis_qids = cluster_info[best_dbname]['qids']
            vis_inliers = []  # np.zeros(shape=(vis_matches.shape[0], 1), dtype=np.int) - 1
            for idx, vid in enumerate(vis_qids):
                if vis_p3d_ids[idx] == q_p3d_ids[vid]:
                    vis_inliers.append(True)
                else:
                    vis_inliers.append(False)
            vis_inliers = np.array(vis_inliers, np.bool).reshape(-1, 1)

            show_proj = True
            if show_proj:
                matched_points2Ddb = [vis_mkpdb[i] for i in range(len(vis_inliers)) if vis_inliers[i]]
                matched_points2D = [vis_mkpq[i] for i in range(len(vis_inliers)) if vis_inliers[i]]
                matched_points3D = [vis_mp3d[i] for i in range(len(vis_inliers)) if vis_inliers[i]]
                matched_points2Ddb = np.vstack(matched_points2Ddb)
                matched_points2D = np.vstack(matched_points2D)
                matched_points3D = np.vstack(matched_points3D)

                reproj_points2D = reproject(points3D=matched_points3D, rvec=ret['qvec'], tvec=ret['tvec'],
                                            camera=cfg)
                proj_error = (matched_points2D - reproj_points2D) ** 2
                proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])

                min_proj_error = np.min(proj_error)
                max_proj_error = np.max(proj_error)
                med_proj_error = np.median(proj_error)
                # print('proj_error: ',  np.max(proj_error))

                img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_points2D, reproj_points2D=reproj_points2D)
                img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
                img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 255, 0), 2)
                img_proj = cv2.putText(img_proj,
                                       'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                              max_proj_error),
                                       (20, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 0, 255), 2)

            if global_score is not None:
                if best_dbname in global_score.keys():
                    gscore = global_score[best_dbname]
                else:
                    gscore = 0
            else:
                gscore = 0

            text = qname + '_' + best_dbname

            db_img = cv2.imread(osp.join(image_dir, best_dbname))
            if with_label:
                if seg_dir is not None:
                    # q_seg = cv2.imread(osp.join(seg_dir, qname.replace("jpg", "png")))
                    q_seg = cv2.imread(osp.join(seg_dir, qname))
                    # db_seg = cv2.imread(osp.join(seg_dir, best_dbname.replace("jpg", "png")))
                    db_seg = cv2.imread(osp.join(seg_dir, best_dbname))

                    q_seg = cv2.resize(q_seg, dsize=(q_img.shape[1], q_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    db_seg = cv2.resize(db_seg, dsize=(db_img.shape[1], db_img.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                else:
                    q_seg = None
                    db_seg = None
            else:
                q_seg = None
                db_seg = None

            img_pair = plot_matches(img1=q_img, img2=db_img,
                                    pts1=vis_mkpq, pts2=vis_mkpdb,
                                    inliers=vis_inliers, plot_outlier=False, plot_match=False)
            if q_seg is not None and db_seg is not None:
                img_seg = plot_matches(img1=q_seg, img2=db_seg,
                                       pts1=vis_mkpq, pts2=vis_mkpdb,
                                       inliers=vis_inliers, plot_outlier=False, plot_match=False)

                img_seg = resize_img(img_seg, nh=img_pair.shape[0])
                img_pair = np.hstack([img_pair, img_seg])

            img_match = plot_matches(img1=q_img, img2=db_img,
                                     pts1=vis_mkpq, pts2=vis_mkpdb,
                                     inliers=vis_inliers, plot_outlier=False)
            img_match_ntext = deepcopy(img_match)
            img_pair = np.hstack([img_pair, img_match_ntext])

            img_match = cv2.putText(img_match, 'm/i/r/o:{:d}/{:d}/{:.2f}/{:d}/{:.4f}/{:s}'.format(
                vis_matches.shape[0], best_inliers, best_inliers / vis_matches.shape[0], cluster_idx + 1, gscore,
                ret_source),
                                    (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
            vis_obs = [len(points3D[v].image_ids) for v in vis_p3d_ids]
            mn_obs = np.min(vis_obs)
            md_obs = np.median(vis_obs)
            mx_obs = np.max(vis_obs)
            img_match = cv2.putText(img_match,
                                    'obs: mn/md/mx:{:d}/{:d}/{:d}'.format(int(mn_obs), int(md_obs),
                                                                          int(mx_obs)),
                                    (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

            ref_q_error, ref_t_error, ref_t_error_xyz = compute_pose_error(pred_qcw=ret['qvec'], pred_tcw=ret['tvec'],
                                                                           gt_qcw=db_images[
                                                                               db_name_to_id[best_dbname]].qvec,
                                                                           gt_tcw=db_images[
                                                                               db_name_to_id[best_dbname]].tvec)
            img_match = cv2.putText(img_match,
                                    'w/ref q:{:.2f}deg t:{:.2f}m'.format(ref_q_error, ref_t_error),
                                    (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

            if gt_qvec is not None and gt_tvec is not None:
                gt_reproj_points2D = reproject(points3D=matched_points3D, rvec=gt_qvec, tvec=gt_tvec,
                                               camera=cfg)
                gt_proj_error = (matched_points2D - gt_reproj_points2D) ** 2
                gt_proj_error = np.sqrt(gt_proj_error[:, 0] + gt_proj_error[:, 1])
                gt_min_proj_error = np.min(gt_proj_error)
                gt_max_proj_error = np.max(gt_proj_error)
                gt_med_proj_error = np.median(gt_proj_error)
                img_proj = cv2.putText(img_proj,
                                       'gt-mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(gt_min_proj_error,
                                                                                 gt_med_proj_error,
                                                                                 gt_max_proj_error),
                                       (20, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 0, 255), 2)
                # gt_proj_error = np.sqrt(gt_proj_error[:, 0] + gt_proj_error[:, 1])
                gt_inliers = []
                for i in range(gt_proj_error.shape[0]):
                    if gt_proj_error[i] <= thresh:
                        gt_inliers.append(True)
                    else:
                        gt_inliers.append(False)

                gt_inliers = np.array(gt_inliers, np.bool).reshape(-1, 1)
                gt_img_match = plot_matches(img1=q_img, img2=db_img,
                                            pts1=matched_points2D, pts2=matched_points2Ddb,
                                            inliers=gt_inliers, plot_outlier=True)
                gt_img_match = cv2.putText(gt_img_match, 'gt-m/i/r/o:{:d}/{:d}/{:.2f}/{:d}/{:.4f}/{:s}'.format(
                    vis_matches.shape[0], np.sum(gt_inliers), np.sum(gt_inliers) / gt_proj_error.shape[0],
                                                              cluster_idx + 1, gscore, ret_source),
                                           (20, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                q_error, t_error, t_error_xyz = compute_pose_error(pred_qcw=ret['qvec'],
                                                                   pred_tcw=ret['tvec'],
                                                                   gt_qcw=gt_qvec,
                                                                   gt_tcw=gt_tvec)
                gt_img_match = cv2.putText(gt_img_match,
                                           'gt-q_err:{:.2f}deg t_err:{:.2f}m'.format(q_error, t_error),
                                           (20, 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)
                gt_img_match = cv2.putText(gt_img_match,
                                           'gt-tx:{:.2f} ty:{:.2f} tz:{:.2f}'.format(t_error_xyz[0], t_error_xyz[1],
                                                                                     t_error_xyz[2]),
                                           (20, 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                img_match = np.hstack([img_match, gt_img_match])

            img_pair = resize_img(img_pair, nh=img_match.shape[0])
            img_match = np.hstack([img_pair, img_match])
            img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)

            if q_seg is not None and db_seg is not None:
                q_seg = cv2.resize(q_seg, dsize=(q_img.shape[1], q_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                db_seg = cv2.resize(db_seg, dsize=(db_img.shape[1], db_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                seg_match = plot_matches(img1=q_seg, img2=db_seg,
                                         pts1=vis_mkpq, pts2=vis_mkpdb,
                                         inliers=vis_inliers)
                seg_match = cv2.resize(seg_match, None, fx=0.5, fy=0.5)
                # img_match = np.hstack([img_match, seg_match])

            if show_proj:
                img_proj = resize_img(img_proj, nh=img_match.shape[0])
                # img_match = np.hstack([img_match, img_proj])
                # print('img_match: ', img_match.shape, img_proj.shape)

            # cv2.imshow("match", img_match)

            key = cv2.waitKey(5)
            if vis_dir is not None:
                id_str = '{:03d}'.format(cluster_idx + 1)
                cv2.imwrite(osp.join(vis_dir.as_posix(),
                                     (qname.replace('/', '-') + '_' + id_str + '_' + best_dbname.replace('/', '-'))),
                            img_match)
        keep = False
        # if best_inliers >= 10:
        #     if ret['num_inliers'] > best_results['num_inliers']:
        #         keep = True
        #     else:
        #         if best_results['single_num_inliers'] < 10:
        #             keep = True
        # else:
        if best_inliers < 8:  # at least 8 inliers from a single image
            keep = False
        elif ret['num_inliers'] <= best_results['num_inliers']:
            keep = False
        else:
            keep = True
        if keep:
            best_results['qvec'] = ret['qvec']
            best_results['tvec'] = ret['tvec']
            best_results['inlier'] = ret['inliers']
            best_results['num_inliers'] = ret['num_inliers']
            best_results['single_num_inliers'] = best_inliers
            best_results['dbname'] = best_dbname
            best_results['order'] = cluster_idx + 1
            best_results['ret_source'] = ret_source

        if ret['num_inliers'] < inlier_th or best_inliers < 10:
            print_text = "qname: {:s} dbname: {:s} ({:s} {:d}/{:d}) failed insufficient {:d}/{:d} inliers".format(qname,
                                                                                                                  best_dbname,
                                                                                                                  ret_source,
                                                                                                                  cluster_idx + 1,
                                                                                                                  len(
                                                                                                                      db_ids),
                                                                                                                  best_inliers,
                                                                                                                  ret[
                                                                                                                      "num_inliers"])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')

            continue

        if not keep:
            best_results['qvec'] = ret['qvec']
            best_results['tvec'] = ret['tvec']
            best_results['inlier'] = ret['inliers']
            best_results['num_inliers'] = ret['num_inliers']
            best_results['single_num_inliers'] = best_inliers
            best_results['dbname'] = best_dbname
            best_results['order'] = cluster_idx + 1
            best_results['ret_source'] = ret_source

        if do_cluster_check:
            cluster = is_cluster(H=cfg['height'], W=cfg['width'], points2D=matched_points2D, radius=50, ratio=0.6)
            if cluster:
                print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed because of cluster".format(qname,
                                                                                                     db_name,
                                                                                                     cluster_idx + 1,
                                                                                                     len(db_ids))
                print(print_text)
                if log_info is not None:
                    log_info += (print_text + '\n')

        print_text = "qname: {:s} dbname: {:s} ({:s} {:d}/{:d}) initialization succeed with {:d}/{:d} inliers".format(
            qname,
            best_dbname,
            ret_source,
            cluster_idx + 1,
            len(db_ids),
            best_inliers,
            ret["num_inliers"]
        )
        print(print_text)
        if log_info is not None:
            log_info += (print_text + '\n')

        if do_covisility_opt:
            if opt_type.find("proj") >= 0:
                q_p3d_ids = {}
                for i in range(len(ret['inliers'])):
                    if ret['inliers'][i]:
                        q_p3d_ids[q_ids[i]] = mp3d_ids[i]
                ret = pose_refinment_covisibility_by_projection(qname=qname, cfg=cfg, feature_file=feature_file,
                                                                db_frame_id=db_name_to_id[best_dbname],
                                                                db_images=db_images, points3D=points3D,
                                                                thresh=thresh, with_label=with_label,
                                                                covisibility_frame=covisibility_frame,
                                                                matcher=matcher,
                                                                # ref_3Dpoints=inlier_p3d_ids,
                                                                ref_3Dpoints=None,
                                                                q_p3d_ids=q_p3d_ids,
                                                                radius=radius,
                                                                qvec=ret['qvec'],
                                                                tvec=ret['tvec'],
                                                                n_can_3Ds=30,
                                                                plus05=plus05,
                                                                iters=iters,
                                                                opt_type=opt_type,
                                                                opt_th=opt_th,
                                                                obs_th=obs_th,
                                                                image_dir=image_dir,
                                                                with_dist=with_dist,
                                                                log_info='',
                                                                )
            elif opt_type.find('clu') >= 0:
                ret = pose_refinement_covisibility(qname=qname,
                                                   cfg=cfg,
                                                   feature_file=feature_file,
                                                   db_frame_id=db_name_to_id[best_dbname],
                                                   db_images=db_images, points3D=points3D,
                                                   thresh=thresh, with_label=with_label,
                                                   covisibility_frame=covisibility_frame,
                                                   matcher=matcher,
                                                   # ref_3Dpoints=inlier_p3d_ids,
                                                   ref_3Dpoints=None,
                                                   plus05=plus05,
                                                   iters=iters,
                                                   obs_th=obs_th,
                                                   opt_th=opt_th,
                                                   radius=radius,
                                                   qvec=ret['qvec'],
                                                   tvec=ret['tvec'],
                                                   log_info='',
                                                   opt_type=opt_type,
                                                   image_dir=image_dir,
                                                   vis_dir=vis_dir,
                                                   depth_th=depth_th,
                                                   gt_qvec=gt_qvec,
                                                   gt_tvec=gt_tvec,
                                                   )
            log_info = log_info + ret['log_info']
            print_text = 'Find {:d} inliers after optimization'.format(ret['num_inliers'])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + "\n")

            if not ret['success']:
                continue

            show_refinment = True
            if show_refinment:
                all_ref_db_ids = ret['db_ids']
                all_mkpq = ret['mkpq']
                all_3D_ids = ret['3D_ids']
                inlier_mask = ret['inliers']

                dbname_ninliers = {}
                dbname_matches = {}
                for did in all_ref_db_ids:
                    db_p3D_ids = db_images[did].point3D_ids
                    dbname = db_images[did].name
                    mkdb = feature_file[dbname]['keypoints'].__array__()

                    matched_mkpq = []
                    matched_p3d = []
                    matched_mkdb = []
                    matched_obs = []
                    for pi in range(len(all_3D_ids)):
                        if not inlier_mask[pi]:
                            continue
                        p3D_id = all_3D_ids[pi]
                        if p3D_id in db_p3D_ids:
                            matched_mkpq.append(all_mkpq[pi])
                            matched_p3d.append(points3D[p3D_id].xyz)
                            mkdb_idx = list(db_p3D_ids).index(p3D_id)
                            matched_mkdb.append(mkdb[mkdb_idx])

                            obs = len(points3D[p3D_id].image_ids)
                            matched_obs.append(obs)

                    if len(matched_p3d) == 0:
                        continue

                    dbname_matches[dbname] = {
                        'mkpq': np.array(matched_mkpq).reshape(-1, 2),
                        'mp3d': np.array(matched_p3d).reshape(-1, 3),
                        'mkpdb': np.array(matched_mkdb).reshape(-1, 2),
                        'min_obs': np.min(matched_obs),
                        'median_obs': np.median(matched_obs),
                        'max_obs': np.max(matched_obs),
                        'all_obs': matched_obs,
                    }
                    dbname_ninliers[dbname] = len(matched_p3d)

                sorted_dbname_ninliers = sort_dict_by_value(data=dbname_ninliers, reverse=True)

                for idx, item in enumerate(sorted_dbname_ninliers):
                    if item[1] == 0:
                        continue

                    dbname = item[0]
                    db_img = cv2.imread(osp.join(image_dir, dbname))

                    matched_mkpq = dbname_matches[dbname]['mkpq']
                    matched_mkdb = dbname_matches[dbname]['mkpdb']
                    matched_p3d = dbname_matches[dbname]['mp3d']
                    reproj_mkpq = reproject(points3D=matched_p3d, rvec=ret['qvec'], tvec=ret['tvec'], camera=cfg)
                    proj_error = (matched_mkpq - reproj_mkpq) ** 2
                    proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
                    min_proj_error = np.min(proj_error)
                    max_proj_error = np.max(proj_error)
                    med_proj_error = np.median(proj_error)
                    img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_mkpq, reproj_points2D=reproj_mkpq)
                    img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
                    img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 255, 0), 2)
                    img_proj = cv2.putText(img_proj,
                                           'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                                  max_proj_error),
                                           (20, 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                    img_match = plot_matches(img1=q_img, img2=db_img, pts1=matched_mkpq, pts2=matched_mkdb,
                                             inliers=np.array([True for i in range(matched_mkpq.shape[0])], np.uint8))
                    img_match = cv2.putText(img_match, 'i/o:{:d}/{:d}'.format(matched_mkpq.shape[0], idx + 1),
                                            (50, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2
                                            )

                    mn_obs = dbname_matches[dbname]['min_obs']
                    md_obs = dbname_matches[dbname]['median_obs']
                    mx_obs = dbname_matches[dbname]['max_obs']
                    img_match = cv2.putText(img_match,
                                            'obs-mn/md/mx:{:d}/{:d}/{:d}'.format(int(mn_obs),
                                                                                 int(md_obs),
                                                                                 int(mx_obs)),
                                            (20, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)
                    img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)
                    img_proj = resize_img(img_proj, nh=img_match.shape[0])
                    img_match = np.hstack([img_match, img_proj])

                    cv2.imshow('match_ref', img_match)
                    key = cv2.waitKey(5)
                    if vis_dir is not None:
                        id_str = '{:03d}'.format(idx + 1)
                        cv2.imwrite(osp.join(vis_dir.as_posix(),
                                             (qname.replace('/',
                                                            '-') + '_' + opt_type + id_str + '_' + dbname.replace(
                                                 '/',
                                                 '-'))),
                                    img_match)

        # localization succeed
        qvec = ret['qvec']
        tvec = ret['tvec']
        ret['cfg'] = cfg
        num_inliers = ret['num_inliers']

        # del cluster_info
        # del dbname_matches
        # del ret
        # del matched_mkpq
        # del matched_p3d
        # del matched_mkdb
        # del matched_points2D
        # del matched_points3D

        return qvec, tvec, num_inliers, {**best_results, **{'log_info': log_info}}

    if best_results['num_inliers'] >= 10:  # 20 for aachen
        qvec = best_results['qvec']
        tvec = best_results['tvec']
        num_inliers = best_results['num_inliers']
        best_dbname = best_results['dbname']
        inliers = best_results['inliers']

        if do_covisility_opt:
            if opt_type.find("proj") >= 0:
                q_p3d_ids = {}
                for i in range(len(inliers)):
                    if inliers[i]:
                        q_p3d_ids[q_ids[i]] = mp3d_ids[i]
                ret = pose_refinment_covisibility_by_projection(qname=qname, cfg=cfg, feature_file=feature_file,
                                                                db_frame_id=db_name_to_id[best_dbname],
                                                                db_images=db_images, points3D=points3D,
                                                                thresh=thresh, with_label=with_label,
                                                                covisibility_frame=covisibility_frame,
                                                                matcher=matcher,
                                                                # ref_3Dpoints=inlier_p3d_ids,
                                                                ref_3Dpoints=None,
                                                                q_p3d_ids=q_p3d_ids,
                                                                radius=radius,
                                                                qvec=qvec,
                                                                tvec=tvec,
                                                                n_can_3Ds=30,
                                                                plus05=plus05,
                                                                iters=iters,
                                                                opt_type=opt_type,
                                                                opt_th=opt_th,
                                                                obs_th=obs_th,
                                                                image_dir=image_dir,
                                                                with_dist=with_dist,
                                                                log_info='',
                                                                )
            elif opt_type.find('clu') >= 0:
                ret = pose_refinement_covisibility(qname=qname,
                                                   cfg=cfg,
                                                   feature_file=feature_file,
                                                   db_frame_id=db_name_to_id[best_dbname],
                                                   db_images=db_images, points3D=points3D,
                                                   thresh=thresh, with_label=with_label,
                                                   covisibility_frame=covisibility_frame,
                                                   matcher=matcher,
                                                   # ref_3Dpoints=inlier_p3d_ids,
                                                   ref_3Dpoints=None,
                                                   plus05=plus05,
                                                   iters=iters,
                                                   obs_th=obs_th,
                                                   opt_th=opt_th,
                                                   radius=radius,
                                                   qvec=qvec,
                                                   tvec=tvec,
                                                   log_info='',
                                                   opt_type=opt_type,
                                                   image_dir=image_dir,
                                                   vis_dir=vis_dir,
                                                   gt_qvec=gt_qvec,
                                                   gt_tvec=gt_tvec,
                                                   )
            log_info = log_info + ret['log_info']
            print_text = 'Find {:d} inliers after optimization'.format(ret['num_inliers'])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + "\n")

            show_refinment = True
            if show_refinment:
                all_ref_db_ids = ret['db_ids']
                all_mkpq = ret['mkpq']
                all_3D_ids = ret['3D_ids']
                inlier_mask = ret['inliers']

                dbname_ninliers = {}
                dbname_matches = {}
                for did in all_ref_db_ids:
                    db_p3D_ids = db_images[did].point3D_ids
                    dbname = db_images[did].name
                    mkdb = feature_file[dbname]['keypoints'].__array__()

                    matched_mkpq = []
                    matched_p3d = []
                    matched_mkdb = []
                    for pi in range(len(all_3D_ids)):
                        if not inlier_mask[pi]:
                            continue
                        p3D_id = all_3D_ids[pi]
                        if p3D_id in db_p3D_ids:
                            matched_mkpq.append(all_mkpq[pi])
                            matched_p3d.append(points3D[p3D_id].xyz)
                            mkdb_idx = list(db_p3D_ids).index(p3D_id)
                            matched_mkdb.append(mkdb[mkdb_idx])

                    if len(matched_p3d) == 0:
                        continue

                    dbname_matches[dbname] = {
                        'mkpq': np.array(matched_mkpq).reshape(-1, 2),
                        'mp3d': np.array(matched_p3d).reshape(-1, 3),
                        'mkpdb': np.array(matched_mkdb).reshape(-1, 2),
                    }
                    dbname_ninliers[dbname] = len(matched_p3d)

                sorted_dbname_ninliers = sort_dict_by_value(data=dbname_ninliers, reverse=True)

                for idx, item in enumerate(sorted_dbname_ninliers):
                    if item[1] == 0:
                        continue

                    dbname = item[0]
                    db_img = cv2.imread(osp.join(image_dir, dbname))

                    matched_mkpq = dbname_matches[dbname]['mkpq']
                    matched_mkdb = dbname_matches[dbname]['mkpdb']
                    matched_p3d = dbname_matches[dbname]['mp3d']
                    reproj_mkpq = reproject(points3D=matched_p3d, rvec=ret['qvec'], tvec=ret['tvec'], camera=cfg)
                    proj_error = (matched_mkpq - reproj_mkpq) ** 2
                    proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
                    min_proj_error = np.min(proj_error)
                    max_proj_error = np.max(proj_error)
                    med_proj_error = np.median(proj_error)
                    img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_mkpq, reproj_points2D=reproj_mkpq)
                    img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
                    img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 255, 0), 2)
                    img_proj = cv2.putText(img_proj,
                                           'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                                  max_proj_error),
                                           (20, 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                    img_match = plot_matches(img1=q_img, img2=db_img, pts1=matched_mkpq, pts2=matched_mkdb,
                                             inliers=np.array([True for i in range(matched_mkpq.shape[0])], np.uint8))
                    img_match = cv2.putText(img_match, 'i/o:{:d}/{:d}'.format(matched_mkpq.shape[0], idx + 1),
                                            (50, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2
                                            )
                    img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)
                    img_proj = resize_img(img_proj, nh=img_match.shape[0])
                    img_match = np.hstack([img_match, img_proj])

                    # cv2.imshow('match_ref', img_match)
                    key = cv2.waitKey(5)
                    if vis_dir is not None:
                        id_str = '{:03d}'.format(idx + 1)
                        cv2.imwrite(osp.join(vis_dir.as_posix(),
                                             (qname.replace('/',
                                                            '-') + '_' + opt_type + id_str + '_' + dbname.replace(
                                                 '/',
                                                 '-'))),
                                    img_match)

        # localization succeed
        qvec = ret['qvec']
        tvec = ret['tvec']
        ret['cfg'] = cfg
        num_inliers = ret['num_inliers']

        # del cluster_info
        # del dbname_matches
        # del ret
        # del matched_mkpq
        # del matched_p3d
        # del matched_mkdb
        # del matched_points2D
        # del matched_points3D

        # return qvec, tvec, num_inliers, {**best_results, **{'log_info': log_info}}
        return qvec, tvec, 0, {**best_results, **{'log_info': log_info}}

        # print_text = 'Localize {:s} failed, but use the estimation from {:s}'.format(qname, best_results['dbname'])
        # print(print_text)
        # if log_info is not None:
        #     log_info += (print_text + '\n')
        # return qvec, tvec, num_inliers, {**best_results, **{'log_info': log_info}}

    db_largest_score = -1
    db_largest_score_cluster_id = None
    for cluster_idx, db_id_cls in enumerate(db_ids):
        mean_score = 0
        for db_id in db_id_cls:
            db_name = db_images[db_id].name
            score = global_score[db_name]
            mean_score += score

        mean_score = mean_score / len(db_id_cls)
        # print(mean_score, db_largest_score)
        if mean_score > db_largest_score:
            db_largest_score = mean_score
            db_largest_score_cluster_id = cluster_idx

    closest = db_images[db_ids[db_largest_score_cluster_id][0]]
    # closest = db_images[db_ids[0][0]]
    print_text = 'Localize {:s} failed, but use the pose of {:s}/{:.2f} as approximation'.format(qname, closest.name,
                                                                                                 db_largest_score)
    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')
    return closest.qvec, closest.tvec, -1, {**best_results, **{'log_info': log_info}}


def pose_from_matcher_hloc(qname, qinfo, db_ids, db_images, points3D,
                           feature_file,
                           thresh,
                           image_dir,
                           matcher,
                           do_covisility_opt=False,
                           with_label=False,
                           vis_dir=None,
                           inlier_th=10,
                           covisibility_frame=50,
                           global_score=None,
                           seg_dir=None,
                           q_seg=None,
                           log_info=None,
                           opt_type="cluster",
                           plus05=False,
                           do_cluster_check=False,
                           iters=1,
                           radius=12, ):
    def do_covisibility_clustering(frame_ids, all_images, points3D):
        clusters = []
        visited = set()
        for frame_id in frame_ids:
            # Check if already labeled
            if frame_id in visited:
                continue
            # New component
            clusters.append([])
            queue = {frame_id}
            while len(queue):
                exploration_frame = queue.pop()
                # Already part of the component
                if exploration_frame in visited:
                    continue
                visited.add(exploration_frame)
                clusters[-1].append(exploration_frame)

                observed = all_images[exploration_frame].point3D_ids
                connected_frames = set(
                    j for i in observed if i != -1 for j in points3D[i].image_ids)
                connected_frames &= set(frame_ids)
                connected_frames -= visited
                queue |= connected_frames

        clusters = sorted(clusters, key=len, reverse=True)
        return clusters

    print("qname: ", qname)
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    kpq = feature_file[qname]['keypoints'].__array__()
    desc_q = feature_file[qname]['descriptors'].__array__()
    desc_q = desc_q.transpose()

    if with_label:
        label_q = feature_file[qname]['labels'].__array__()
    else:
        label_q = None

    # results = {}
    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }

    best_results = {
        'tvec': None,
        'rvec': None,
        'num_inliers': 0,
        'db_id': -1,
        'order': -1,
        'qname': qname,
        'dbname': db_images[db_ids[0]].name,
    }

    if do_covisility_opt:
        clusters = do_covisibility_clustering(frame_ids=db_ids, all_images=db_images, points3D=points3D)
        print("Find {:d} clusters".format(len(clusters)))
        best_inliers = 0
        best_cluster = None
        best_qvec = None
        best_tvec = None

        for i, cluster_ids in enumerate(clusters):
            cluster_info, mp3d, mkpq, mp3d_ids, q_ids = match_cluster_2D(kpq=kpq, desc_q=desc_q, label_q=label_q,
                                                                         db_ids=cluster_ids,
                                                                         points3D=points3D,
                                                                         feature_file=feature_file,
                                                                         db_images=db_images,
                                                                         with_label=with_label,
                                                                         matcher=matcher,
                                                                         plus05=True,
                                                                         )
            ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)
            if not ret['success']:
                print('Localization failed with cluster {:d}/{:d}'.format(i + 1, len(clusters)))
            else:
                print('Find {:d} inliers from cluster {:d}/{:d} of {:d} frames'.format(ret['num_inliers'], i + 1,
                                                                                       len(clusters),
                                                                                       len(cluster_info.keys())))

            if ret['success'] and ret['num_inliers'] > best_inliers:
                best_cluster = i
                best_inliers = ret['num_inliers']
                best_qvec = ret['qvec']
                best_tvec = ret['tvec']

        if best_cluster is not None:
            print('Find best cluster {:d} with {:d} inliers'.format(best_cluster + 1, best_inliers))
            return best_qvec, best_tvec, best_inliers, best_results
    else:
        cluster_info, mp3d, mkpq, mp3d_ids, q_ids = match_cluster_2D(kpq=kpq, desc_q=desc_q, label_q=label_q,
                                                                     db_ids=db_ids,
                                                                     points3D=points3D,
                                                                     feature_file=feature_file,
                                                                     db_images=db_images,
                                                                     with_label=with_label,
                                                                     matcher=matcher,
                                                                     plus05=True,
                                                                     )
        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)

        if ret['success']:
            return ret['qvec'], ret['tvec'], ret['num_inliers'], best_results

    closest = db_images[db_ids[0]]
    return closest.qvec, closest.tvec, 0, best_results


def pose_from_cluster_with_matcher_hloc(qname, qinfo, db_ids, db_images, points3D,
                                        cameras,
                                        feature_file,
                                        thresh,
                                        image_dir,
                                        matcher,
                                        do_covisility_opt=False,
                                        with_label=False,
                                        vis_dir=None,
                                        inlier_th=10,
                                        covisibility_frame=50,
                                        global_score=None,
                                        seg_dir=None,
                                        q_seg=None,
                                        log_info=None,
                                        opt_type="cluster",
                                        plus05=False,
                                        do_cluster_check=False,
                                        iters=1,
                                        radius=12,
                                        obs_th=0,
                                        opt_th=12,
                                        with_dist=False,
                                        ):
    print("qname: ", qname)
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    kpq = feature_file[qname]['keypoints'].__array__()
    desc_q = feature_file[qname]['descriptors'].__array__()
    # score_q = feature_file[qname]['score'].__array__()
    desc_q = desc_q.transpose()

    if with_label:
        label_q = feature_file[qname]['labels'].__array__()
    else:
        label_q = None

    # results = {}
    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }

    best_results = {
        'tvec': None,
        'rvec': None,
        'num_inliers': 0,
        'db_id': -1,
        'order': -1,
        'qname': qname,
        'dbname': db_images[db_ids[0][0]].name,
    }

    for cluster_idx, db_id_cls in enumerate(db_ids):
        db_id = db_id_cls[0]
        db_name = db_images[db_id].name
        cluster_info, mp3d, mkpq, mp3d_ids, q_ids = match_cluster_2D(kpq=kpq, desc_q=desc_q, label_q=label_q,
                                                                     db_ids=db_id_cls,
                                                                     points3D=points3D,
                                                                     feature_file=feature_file,
                                                                     db_images=db_images,
                                                                     with_label=with_label,
                                                                     matcher=matcher,
                                                                     plus05=plus05,
                                                                     # obs_th=obs_th,
                                                                     obs_th=3,
                                                                     )
        if mp3d.shape[0] < inlier_th:
            print_text = "qname: {:s} dbname: {:s}({:d}/{:d}) failed because of insufficient 3d points {:d}".format(
                qname,
                db_name,
                cluster_idx + 1,
                len(
                    db_ids),
                mp3d.shape[
                    0])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)
        # print('ret after initialization: ', ret)
        # exit(0)

        if not ret["success"]:
            # results.append((qname, db_name, 0))
            print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed after initialization".format(qname, db_name,
                                                                                                   cluster_idx + 1,
                                                                                                   len(db_ids))
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        # with open('pairs-netvlad50-inliers-500-1015.txt', 'a+') as f:
        #     if not ret['success']:
        #         text = "{:s} {:s} with {:d} inliers".format(qname, db_name, 0)
        #     else:
        #         text = "{:s} {:s} with {:d} inliers".format(qname, db_name, ret['num_inliers'])
        #     f.write(text + '\n')
        #
        #     print(text)
        #
        # if cluster_idx == len(db_ids) - 1:
        #     return ret['qvec'], ret['tvec'], ret['num_inliers'], best_results
        # else:
        #     continue

        inliers = ret['inliers']
        inlier_p3d_ids = [mp3d_ids[i] for i in range(len(inliers)) if inliers[i]]

        # visualize the matches
        q_p3d_ids = np.zeros(shape=(desc_q.shape[0], 1), dtype=np.int) - 1
        for idx, qid in enumerate(q_ids):
            if inliers[idx]:
                q_p3d_ids[qid] = mp3d_ids[idx]

        best_dbname = None
        best_inliers = -1
        for db_name in cluster_info.keys():
            matched_mp3d_ids = cluster_info[db_name]['mp_3d_ids']
            matched_qids = cluster_info[db_name]['qids']
            n = 0
            for idx, qid in enumerate(matched_qids):
                if matched_mp3d_ids[idx] == q_p3d_ids[qid]:
                    n += 1
            if n > best_inliers:
                best_inliers = n
                best_dbname = db_name

        if best_dbname is not None:
            # print('best_dbname: ', best_dbname)
            # '''
            vis_matches = cluster_info[best_dbname]['matches']
            vis_p3d_ids = cluster_info[best_dbname]['mp_3d_ids']
            vis_mkpdb = cluster_info[best_dbname]['mkpdb']
            vis_mkpq = cluster_info[best_dbname]['mkpq']
            vis_mp3d = cluster_info[best_dbname]['mp3d']
            vis_qids = cluster_info[best_dbname]['qids']
            vis_inliers = []  # np.zeros(shape=(vis_matches.shape[0], 1), dtype=np.int) - 1
            for idx, vid in enumerate(vis_qids):
                if vis_p3d_ids[idx] == q_p3d_ids[vid]:
                    vis_inliers.append(True)
                else:
                    vis_inliers.append(False)
            vis_inliers = np.array(vis_inliers, np.bool).reshape(-1, 1)
            matched_points2D = [vis_mkpq[i] for i in range(len(vis_inliers)) if vis_inliers[i]]
            matched_points3D = [vis_mp3d[i] for i in range(len(vis_inliers)) if vis_inliers[i]]
            matched_points2D = np.vstack(matched_points2D)
            matched_points3D = np.vstack(matched_points3D)
            show_proj = False
            if show_proj:
                reproj_points2D = reproject(points3D=matched_points3D, rvec=ret['qvec'], tvec=ret['tvec'],
                                            camera=cfg)

                proj_error = (matched_points2D - reproj_points2D) ** 2
                proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
                min_proj_error = np.min(proj_error)
                max_proj_error = np.max(proj_error)
                med_proj_error = np.median(proj_error)
                # print('proj_error: ',  np.max(proj_error))

                img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_points2D, reproj_points2D=reproj_points2D)
                img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
                img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 255, 0), 2)
                img_proj = cv2.putText(img_proj,
                                       'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                              max_proj_error),
                                       (20, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 0, 255), 2)

            if global_score is not None:
                if best_dbname in global_score.keys():
                    gscore = global_score[best_dbname]
                else:
                    gscore = 0
            else:
                gscore = 0

            text = qname + '_' + best_dbname

            q_img = cv2.imread(osp.join(image_dir, qname))
            db_img = cv2.imread(osp.join(image_dir, best_dbname))
            # db_img = None
            if with_label:
                if seg_dir is not None:
                    db_seg = cv2.imread(osp.join(seg_dir, best_dbname.replace("jpg", "png")))
                else:
                    db_seg = None
            else:
                db_seg = None
            img_match = plot_matches(img1=q_img, img2=db_img,
                                     pts1=vis_mkpq, pts2=vis_mkpdb,
                                     inliers=vis_inliers, plot_outlier=False)
            img_match = cv2.putText(img_match, 'm/i/r/o:{:d}/{:d}/{:.2f}/{:d}/{:.4f}'.format(
                vis_matches.shape[0], best_inliers, best_inliers / vis_matches.shape[0], cluster_idx + 1, gscore),
                                    (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

            vis_obs = [len(points3D[v].image_ids) for v in vis_p3d_ids]
            mn_obs = np.min(vis_obs)
            md_obs = np.median(vis_obs)
            mx_obs = np.max(vis_obs)
            img_match = cv2.putText(img_match,
                                    'obs: mn/md/mx:{:d}/{:d}/{:d}'.format(int(mn_obs), int(md_obs),
                                                                          int(mx_obs)),
                                    (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

            img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)
            if q_seg is not None and db_seg is not None:
                q_seg = cv2.resize(q_seg, dsize=(q_img.shape[1], q_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                db_seg = cv2.resize(db_seg, dsize=(db_img.shape[1], db_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                seg_match = plot_matches(img1=q_seg, img2=db_seg,
                                         pts1=vis_mkpq, pts2=vis_mkpdb,
                                         inliers=vis_inliers)
                seg_match = cv2.resize(seg_match, None, fx=0.5, fy=0.5)
                img_match = np.hstack([img_match, seg_match])

            if show_proj:
                img_proj = resize_img(img_proj, nh=img_match.shape[0])
                img_match = np.hstack([img_match, img_proj])
            # print('img_match: ', img_match.shape, img_proj.shape)
            cv2.imshow("match", img_match)
            key = cv2.waitKey(5)
            if vis_dir is not None:
                id_str = '{:03d}'.format(cluster_idx + 1)
                cv2.imwrite(osp.join(vis_dir.as_posix(),
                                     (qname.replace('/', '-') + '_' + id_str + '_' + best_dbname.replace('/', '-'))),
                            img_match)
            # '''

            print_text = 'Localize {:s} succeed with best reference {:s}/{:d}'.format(qname, best_dbname, best_inliers)
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')

            if ret['num_inliers'] > best_results['num_inliers']:
                best_results['qvec'] = ret['qvec']
                best_results['tvec'] = ret['qvec']
                best_results['num_inliers'] = ret['num_inliers']
                best_results['dbname'] = best_dbname
                best_results['order'] = cluster_idx + 1

        # calc_uncertainty(mkp=matched_points2D, mp3d=matched_points3D, qvec=ret['qvec'], tvec=ret['tvec'],
        #                  camera=cfg,
        #                  # r_samples=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        #                  r_samples=[0],
        #                  # t_samples=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
        #                  t_samples=np.arange(-1., 1., 0.1),
        #                  )
        # exit(0)

        if ret['num_inliers'] >= inlier_th:
            show_refinement = False
            # if do_covisility_opt and inlier_th < 100:
            if do_covisility_opt:
                show_refinement = True
                q_p3d_ids = {}
                for i in range(len(ret['inliers'])):
                    if ret['inliers'][i]:
                        q_p3d_ids[q_ids[i]] = mp3d_ids[i]
                if opt_type.find('proj') >= 0:
                    ret = pose_refinment_covisibility_by_projection(qname=qname, cfg=cfg, feature_file=feature_file,
                                                                    db_frame_id=db_name_to_id[best_dbname],
                                                                    db_images=db_images, points3D=points3D,
                                                                    thresh=thresh, with_label=with_label,
                                                                    covisibility_frame=covisibility_frame,
                                                                    matcher=matcher,
                                                                    # ref_3Dpoints=inlier_p3d_ids,
                                                                    ref_3Dpoints=None,
                                                                    q_p3d_ids=q_p3d_ids,
                                                                    radius=radius,
                                                                    qvec=ret['qvec'],
                                                                    tvec=ret['tvec'],
                                                                    n_can_3Ds=30,
                                                                    plus05=plus05,
                                                                    iters=iters,
                                                                    log_info='',
                                                                    opt_type=opt_type,
                                                                    obs_th=obs_th,
                                                                    opt_th=opt_th,
                                                                    image_dir=image_dir,
                                                                    with_dist=with_dist,
                                                                    )
                # elif opt_type == 'cluster':
                else:
                    ret = pose_refinement_covisibility(qname=qname,
                                                       cfg=cfg,
                                                       feature_file=feature_file,
                                                       db_frame_id=db_name_to_id[best_dbname],
                                                       db_images=db_images, points3D=points3D,
                                                       thresh=thresh, with_label=with_label,
                                                       covisibility_frame=covisibility_frame,
                                                       matcher=matcher,
                                                       # ref_3Dpoints=inlier_p3d_ids,
                                                       ref_3Dpoints=None,
                                                       plus05=plus05,
                                                       iters=iters,
                                                       obs_th=obs_th,
                                                       opt_th=opt_th,
                                                       radius=radius,
                                                       qvec=ret['qvec'],
                                                       tvec=ret['tvec'],
                                                       log_info=log_info,
                                                       image_dir=image_dir,
                                                       vis_dir=vis_dir,
                                                       )

                log_info = log_info + ret['log_info']
                print_text = 'Find {:d} inliers after optimization'.format(ret['num_inliers'])
                print(print_text)
                if log_info is not None:
                    log_info += (print_text + "\n")

                if not ret['success']:
                    continue

            # show_refinment = True
            if show_refinement:
                all_ref_db_ids = ret['db_ids']
                all_mkpq = ret['mkpq']
                all_score_q = ret['score_q']
                all_3D_ids = ret['3D_ids']
                inlier_mask = ret['inliers']
                q_img = cv2.imread(osp.join(image_dir, qname))

                dbname_ninliers = {}
                dbname_matches = {}
                for did in all_ref_db_ids:
                    db_p3D_ids = db_images[did].point3D_ids
                    dbname = db_images[did].name
                    # if dbname not in ['db/1966.jpg']:
                    #     continue

                    mkdb = feature_file[dbname]['keypoints'].__array__()

                    matched_scoreq = []
                    matched_mkpq = []
                    matched_p3d = []
                    matched_mkdb = []
                    # min_obs = 100000
                    # max_obs = 0
                    matched_obs = []
                    for pi in range(len(all_3D_ids)):
                        if not inlier_mask[pi]:
                            continue
                        p3D_id = all_3D_ids[pi]
                        if p3D_id in db_p3D_ids:
                            matched_scoreq.append(all_score_q[pi])
                            matched_mkpq.append(all_mkpq[pi])
                            matched_p3d.append(points3D[p3D_id].xyz)
                            mkdb_idx = list(db_p3D_ids).index(p3D_id)
                            matched_mkdb.append(mkdb[mkdb_idx])

                            obs = len(points3D[p3D_id].image_ids)
                            matched_obs.append(obs)
                            # if obs < min_obs:
                            #     min_obs = obs
                            # if obs > max_obs:
                            #     max_obs = obs

                            # print ('p3d_id/db_p3d_id: ', p3D_id, db_p3D_ids[mkdb_idx])
                    if len(matched_p3d) == 0:
                        continue

                    dbname_matches[dbname] = {
                        'mkpq': np.array(matched_mkpq).reshape(-1, 2),
                        'mp3d': np.array(matched_p3d).reshape(-1, 3),
                        'mkpdb': np.array(matched_mkdb).reshape(-1, 2),
                        'min_obs': np.min(matched_obs),
                        'median_obs': np.median(matched_obs),
                        'max_obs': np.max(matched_obs),
                        'all_obs': matched_obs,
                        'score_q': np.array(matched_scoreq, np.float32) + 1.,
                    }
                    dbname_ninliers[dbname] = len(matched_p3d)

                sorted_dbname_ninliers = sort_dict_by_value(data=dbname_ninliers, reverse=True)

                for idx, item in enumerate(sorted_dbname_ninliers):

                    dbname = item[0]

                    # if dbname not in ['db/4223.jpg']:
                    #     continue

                    db_img = cv2.imread(osp.join(image_dir, dbname))

                    matched_mkpq = dbname_matches[dbname]['mkpq']
                    matched_scoreq = dbname_matches[dbname]['score_q']
                    matched_p3d = dbname_matches[dbname]['mp3d']
                    reproj_mkpq = reproject(points3D=matched_p3d, rvec=ret['qvec'], tvec=ret['tvec'], camera=cfg)
                    proj_error = (matched_mkpq - reproj_mkpq) ** 2
                    proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
                    min_proj_error = np.min(proj_error)
                    max_proj_error = np.max(proj_error)
                    med_proj_error = np.median(proj_error)
                    img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_mkpq, reproj_points2D=reproj_mkpq,
                                                  # confs=matched_scoreq * 5,
                                                  )
                    img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
                    img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 255, 0), 2)
                    img_proj = cv2.putText(img_proj,
                                           'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                                  max_proj_error),
                                           (20, 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                    #### plot reprojection error on db images
                    # matched_mkpq = dbname_matches[dbname]['mkpq']
                    matched_mkdb = dbname_matches[dbname]['mkpdb']
                    matched_obs = np.array(dbname_matches[dbname]['all_obs'], np.int) // 2
                    # matched_p3d = dbname_matches[dbname]['mp3d']
                    # camera_model, width, height, params = qinfo
                    db_camera = cameras[db_images[db_name_to_id[dbname]].camera_id]
                    db_cfg = {
                        'model': db_camera[1],
                        'width': db_camera[2],
                        'height': db_camera[3],
                        'params': db_camera[4],
                    }
                    reproj_mkdb = reproject(points3D=matched_p3d, rvec=db_images[db_name_to_id[dbname]].qvec,
                                            tvec=db_images[db_name_to_id[dbname]].tvec,
                                            camera=db_cfg)
                    proj_error_db = (matched_mkdb - reproj_mkdb) ** 2
                    proj_error_db = np.sqrt(proj_error_db[:, 0] + proj_error_db[:, 1])
                    min_proj_error_db = np.min(proj_error_db)
                    max_proj_error_db = np.max(proj_error_db)
                    med_proj_error_db = np.median(proj_error_db)
                    img_proj_db = plot_reprojpoint2D(img=db_img, points2D=matched_mkdb, reproj_points2D=reproj_mkdb,
                                                     # confs=matched_obs,
                                                     )
                    img_proj_db = cv2.resize(img_proj_db, None, fx=0.5, fy=0.5)
                    img_proj_db = cv2.putText(img_proj_db, 'green p2D/red-proj', (20, 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1,
                                              (0, 255, 0), 2)
                    img_proj_db = cv2.putText(img_proj_db,
                                              'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error_db,
                                                                                     med_proj_error_db,
                                                                                     max_proj_error_db),
                                              (20, 60),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1,
                                              (0, 0, 255), 2)

                    img_proj_db = resize_img(img_proj_db, nw=img_proj.shape[1])
                    img_proj = np.vstack([img_proj, img_proj_db])

                    img_match = plot_matches(img1=q_img, img2=db_img, pts1=matched_mkpq, pts2=matched_mkdb,
                                             inliers=np.array([True for i in range(matched_mkpq.shape[0])], np.uint8),
                                             )
                    img_match = cv2.putText(img_match, 'i/o:{:d}/{:d}'.format(matched_mkpq.shape[0], idx + 1),
                                            (20, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2
                                            )
                    mn_obs = dbname_matches[dbname]['min_obs']
                    md_obs = dbname_matches[dbname]['median_obs']
                    mx_obs = dbname_matches[dbname]['max_obs']
                    img_match = cv2.putText(img_match,
                                            'obs-mn/md/mx:{:d}/{:d}/{:d}'.format(int(mn_obs),
                                                                                 int(md_obs),
                                                                                 int(mx_obs)),
                                            (20, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)
                    img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)
                    img_raw = plot_matches(img1=q_img, img2=db_img, pts1=matched_mkpq, pts2=matched_mkdb,
                                           inliers=np.array([], np.uint8))
                    img_raw = cv2.resize(img_raw, None, fx=0.5, fy=0.5)

                    # img_proj = resize_img(img_proj, nh=img_match.shape[0])
                    # img_match = np.hstack([img_raw, img_match])

                    cv2.imshow('match_ref', img_match)
                    key = cv2.waitKey(5)
                    if vis_dir is not None:
                        id_str = '{:03d}'.format(idx + 1)
                        cv2.imwrite(osp.join(vis_dir.as_posix(),
                                             (qname.replace('/',
                                                            '-') + '_' + opt_type + id_str + '_' + dbname.replace(
                                                 '/',
                                                 '-'))),
                                    img_match)

                    #### visualize the 3D map & reference db image
                    show_3D = False
                    if show_3D:
                        print("db_seg_dir: ", osp.join(seg_dir, dbname.replace('jpg', 'png')))
                        db_seg = cv2.imread(osp.join(seg_dir, dbname.replace('jpg', 'png')))
                        if q_seg is not None and db_seg is not None:
                            q_seg = cv2.resize(q_seg, dsize=(q_img.shape[1], q_img.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                            db_seg = cv2.resize(db_seg, dsize=(db_img.shape[1], db_img.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                            seg_match = plot_matches(img1=q_seg, img2=db_seg, pts1=matched_mkpq, pts2=matched_mkdb,
                                                     # inliers=np.array([True for i in range(matched_mkpq.shape[0])], np.uint8),
                                                     inliers=np.array([], np.uint8),
                                                     )
                            seg_match = cv2.resize(seg_match, None, fx=0.5, fy=0.5)
                            img_match = np.hstack([img_raw, seg_match, img_match])
                        Visualize3DMap(ref_db_ids=all_ref_db_ids,
                                       points3D=points3D, db_images=db_images,
                                       q_qvec=ret['qvec'], q_tvec=ret['tvec'],
                                       ref_db_id=db_name_to_id[dbname],
                                       img=img_match)

                        vis_3D = {
                            "img_match": img_match,
                            "all_Ref_db_ids": all_ref_db_ids,
                            "ref_db_id": db_name_to_id[dbname],
                            "q_qvec": ret['qvec'],
                            "q_tvec": ret['tvec'],
                        }
                        np.save(qname.replace('/', '-') + ".npy", vis_3D)
                        break

            return ret['qvec'], ret['tvec'], ret['num_inliers'], {**best_results, **{'log_info': log_info}}

    closest = db_images[db_ids[0][0]]
    print_text = 'Localize {:s} failed, but use the pose of {:s} as approximation'.format(qname, closest.name)
    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')
    return closest.qvec, closest.tvec, 0, {**best_results, **{'log_info': log_info}}


def pose_from_single_with_matcher(qname, qinfo, db_ids, db_images, points3D,
                                  feature_file,
                                  thresh,
                                  image_dir,
                                  matcher,
                                  do_covisility_opt=False,
                                  with_label=False,
                                  vis_dir=None,
                                  inlier_th=10,
                                  covisibility_frame=50,
                                  global_score=None,
                                  seg_dir=None,
                                  q_seg=None,
                                  log_info=None,
                                  opt_type="cluster",
                                  ):
    print("qname: ", qname)
    q_img = cv2.imread(osp.join(image_dir, qname))
    kpq = feature_file[qname]['keypoints'].__array__()
    scoreq = feature_file[qname]['scores'].__array__()
    desc_q = feature_file[qname]['descriptors'].__array__()
    # print('kp: ', kpq.shape, desc_q.shape, scoreq.shape)
    # feature_file[qname].keys()
    # exit(0)
    desc_q = desc_q.transpose()

    if with_label:
        label_q = feature_file[qname]['labels'].__array__()
    else:
        label_q = None

    # results = {}
    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }

    best_results = {
        'tvec': None,
        'rvec': None,
        'num_inliers': 0,
        'db_id': -1,
        'order': -1,
        'qname': qname,
        'dbname': db_images[db_ids[0]].name,
    }
    for i, db_id_cls in enumerate(db_ids):
        if len(db_id_cls) == 1:
            db_id = db_id_cls[0]
        db_name = db_images[db_id].name
        if global_score is not None:
            if db_name in global_score.keys():
                gscore = global_score[db_name]
            else:
                gscore = 0
        else:
            gscore = 0

        text = qname + '_' + db_name

        db_img = cv2.imread(osp.join(image_dir, db_name))

        kpdb = feature_file[db_name]['keypoints'].__array__()
        desc_db = feature_file[db_name]["descriptors"].__array__()
        desc_db = desc_db.transpose()

        points3D_ids = db_images[db_id].point3D_ids
        if points3D_ids.size == 0:
            print("No 3D points in this db image: ", db_name)

            if log_info is not None:
                log_info += ("No 3D points in this db image: {:s}\n".format(db_name))
            continue

        if with_label:
            label_db = feature_file[db_name]["labels"].__array__()

            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=label_q, label_db=label_db,
                                       matcher=matcher,
                                       db_3D_ids=points3D_ids,
                                       )

            if seg_dir is not None:
                db_seg = cv2.imread(osp.join(seg_dir, db_name.replace("jpg", "png")))
            else:
                db_seg = None
        else:
            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=None, label_db=None,
                                       matcher=matcher, db_3D_ids=None)
        mp3d = []
        mkpq = []
        mkpdb = []
        mp3d_ids = []
        q_ids = []
        for idx in range(matches.shape[0]):
            if matches[idx] == -1:
                continue
            if points3D_ids[matches[idx]] == -1:
                continue
            id_3D = points3D_ids[matches[idx]]
            mp3d.append(points3D[id_3D].xyz)
            mp3d_ids.append(id_3D)

            mkpq.append(kpq[idx])
            mkpdb.append(kpdb[matches[idx]])

            q_ids.append(idx)

        mp3d = np.array(mp3d, np.float).reshape(-1, 3)
        mkpq = np.array(mkpq, np.float).reshape(-1, 2) + 0.5

        if mp3d.shape[0] < inlier_th:
            print_text = "qname: {:s} dbname: {:s}({:d}/{:d}) failed because of insufficient 3d points {:d}".format(
                qname,
                db_name,
                i + 1,
                len(
                    db_ids),
                mp3d.shape[
                    0])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)

        if not ret["success"]:
            # results.append((qname, db_name, 0))
            print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed after optimization".format(qname, db_name, i + 1,
                                                                                                 len(db_ids))
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')

        ##### plot reprojected 3D points
        show_proj = True
        if show_proj:
            inliers = ret['inliers']
            matched_points2D = [mkpq[i] for i in range(len(inliers)) if inliers[i]]
            matched_points3D = [mp3d[i] for i in range(len(inliers)) if inliers[i]]
            matched_points2D = np.vstack(matched_points2D)
            matched_points3D = np.vstack(matched_points3D)
            reproj_points2D = reproject(points3D=matched_points3D, rvec=ret['qvec'], tvec=ret['tvec'],
                                        camera=cfg)

            proj_error = (matched_points2D - reproj_points2D) ** 2
            proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
            min_proj_error = np.min(proj_error)
            max_proj_error = np.max(proj_error)
            med_proj_error = np.median(proj_error)
            # print('proj_error: ',  np.max(proj_error))

            img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_points2D, reproj_points2D=reproj_points2D)
            img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
            img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 255, 0), 2)
            img_proj = cv2.putText(img_proj,
                                   'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                          max_proj_error),
                                   (20, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 255), 2)

            # img_proj = np.vstack([img_proj, np.zeros_like(img_proj)])
            # img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
            # cv2.imshow('img_proj', img_proj)
            # cv2.waitKey(5)

        matches = np.array(ret['inliers'], np.bool).reshape(-1, 1)
        inlier_p3d_ids = [mp3d_ids[i] for i in range(len(ret['inliers'])) if ret['inliers'][i]]
        # print('matches: ', matches.shape, len(inlier_p3d_ids))
        # print(type(ret['inliers']), ret['inliers'])
        # exit(0)
        img_match = plot_matches(img1=q_img, img2=db_img,
                                 pts1=mkpq, pts2=mkpdb,
                                 inliers=matches)
        img_match = cv2.putText(img_match, 'm/i/r/o:{:d}/{:d}/{:.2f}/{:d}/{:.4f}'.format(
            mkpq.shape[0], ret['num_inliers'], ret['num_inliers'] / mkpq.shape[0], i + 1, gscore), (50, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

        # cv2.namedWindow("match", cv2.WINDOW_NORMAL)

        img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)
        if q_seg is not None and db_seg is not None:
            q_seg = cv2.resize(q_seg, dsize=(q_img.shape[1], q_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            db_seg = cv2.resize(db_seg, dsize=(db_img.shape[1], db_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            seg_match = plot_matches(img1=q_seg, img2=db_seg,
                                     pts1=mkpq, pts2=mkpdb,
                                     inliers=matches)
            seg_match = cv2.resize(seg_match, None, fx=0.5, fy=0.5)
            img_match = np.hstack([img_match, seg_match])

            if show_proj:
                img_proj = resize_img(img_proj, nh=img_match.shape[0])
                img_match = np.hstack([img_match, img_proj])
            # print('img_match: ', img_match.shape, img_proj.shape)
        cv2.imshow("match", img_match)

        key = cv2.waitKey(5)

        if vis_dir is not None:
            id_str = '{:03d}'.format(i + 1)
            cv2.imwrite(osp.join(vis_dir.as_posix(),
                                 (qname.replace('/', '-') + '_' + id_str + '_' + db_name.replace('/', '-'))),
                        img_match)

        if ret['num_inliers'] < inlier_th:
            print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed insufficient {:d} inliers".format(qname, db_name,
                                                                                                        i + 1,
                                                                                                        len(db_ids),
                                                                                                        ret[
                                                                                                            "num_inliers"])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) succeed with {:d} inliers".format(qname, db_name, i + 1,
                                                                                             len(db_ids),
                                                                                             ret["num_inliers"])
        print(print_text)
        if log_info is not None:
            log_info += (print_text + '\n')

        if do_covisility_opt:  # and ret['num_inliers'] <= 200:
            if opt_type == "proj":
                q_p3d_ids = {}
                for i in range(len(ret['inliers'])):
                    if ret['inliers'][i]:
                        q_p3d_ids[q_ids[i]] = mp3d_ids[i]
                ret = pose_refinment_covisibility_by_projection(qname=qname, cfg=cfg, feature_file=feature_file,
                                                                db_frame_id=db_id,
                                                                db_images=db_images, points3D=points3D,
                                                                thresh=thresh, with_label=with_label,
                                                                covisibility_frame=covisibility_frame,
                                                                matcher=matcher,
                                                                ref_3Dpoints=inlier_p3d_ids,
                                                                q_p3d_ids=q_p3d_ids,
                                                                radius=20,
                                                                qvec=ret['qvec'],
                                                                tvec=ret['tvec'],
                                                                n_can_3Ds=50, )
            elif opt_type == "cluster":
                ret = pose_refinement_covisibility(qname=qname, cfg=cfg, feature_file=feature_file, db_frame_id=db_id,
                                                   db_images=db_images,
                                                   points3D=points3D,
                                                   thresh=thresh,
                                                   with_label=with_label,
                                                   covisibility_frame=covisibility_frame,
                                                   matcher=matcher,
                                                   ref_3Dpoints=inlier_p3d_ids,
                                                   )

            if ret['num_inliers'] < 50:
                print_text = 'Find {:d} inliers after covisible refinement, failed'.format(ret['num_inliers'])
                print(print_text)
                if log_info is not None:
                    log_info += (print_text + '\n')
                if ret['num_inliers'] > best_results['num_inliers']:
                    best_results['qvec'] = ret['qvec']
                    best_results['tvec'] = ret['qvec']
                    best_results['num_inliers'] = ret['num_inliers']
                    best_results['dbname'] = db_name
                    best_results['order'] = i + 1

                continue

        print_text = text + "_succeed with {:d} ({:d}/{:d}) inliers".format(ret["num_inliers"], i + 1, len(db_ids))
        print(print_text)
        if log_info is not None:
            log_info += (print_text + '\n')

        # localization succeed
        qvec = ret['qvec']
        tvec = ret['tvec']
        ret['cfg'] = cfg
        inliers = ret['inliers']
        num_inliers = ret['num_inliers']

        if num_inliers > best_results['num_inliers']:
            best_results['qvec'] = ret['qvec']
            best_results['tvec'] = ret['qvec']
            best_results['num_inliers'] = num_inliers
            best_results['dbname'] = db_name
            best_results['order'] = i + 1

        return qvec, tvec, num_inliers, best_results

    if best_results['num_inliers'] >= inlier_th:
        return best_results['qvec'], best_results['tvec'], best_results['num_inliers'], best_results
    closest = db_images[db_ids[0]]
    return closest.qvec, closest.tvec, 0, best_results


def pose_from_pnp(qname, qinfo, db_ids, db_images, points3D,
                  feature_file, match_file, thresh, image_dir):
    print("qname: ", qname)
    q_img = cv2.imread(osp.join(image_dir, qname))
    kpq = feature_file[qname]['keypoints'].__array__()
    param = qinfo[-1]
    f, cx, cy, k = param
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], np.float).reshape(3, 3)
    D = np.array([k, 0, 0, 0], np.float)
    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        print("db_name: ", db_name)

        db_img = cv2.imread(osp.join(image_dir, db_name))

        kpdb = feature_file[db_name]['keypoints'].__array__()

        points3D_ids = db_images[db_id].point3D_ids

        pair = names_to_pair(qname, db_name)
        matches = match_file[pair]['matches0'].__array__()
        valid = np.where(matches > -1)[0]

        if points3D_ids.size == 0:
            print("No 3D points in this db image: ", db_name)
            continue

        valid = valid[points3D_ids[matches[valid]] != -1]

        mp3d = []
        mkpq = []
        mkpdb = []
        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            mp3d.append(points3D[id_3D].xyz)

            mkpq.append(kpq[idx])
            mkpdb.append(kpdb[matches[idx]])

            # p3d_idxes.append(points3D_ids[matches[idx]])

        mp3d = np.array(mp3d, np.float).reshape(-1, 3)
        mkpq = np.array(mkpq, np.float).reshape(-1, 2) + 0.5

        # start to pnp
        # rvec = None
        # tvec = None
        # inliers = None
        if mp3d.shape[0] < 10:
            print("qnqme: {:s} dbname: {:s} failed because of inefficient matches {:d}".format(qname, db_name,
                                                                                               mp3d.shape[0]))
            continue
        success, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=mp3d, imagePoints=mkpq, cameraMatrix=K,
                                                          distCoeffs=D,
                                                          # rvec=rvec,
                                                          # tvec=tvec,
                                                          # inliers=inliers,
                                                          iterationsCount=500,
                                                          reprojectionError=8.0,
                                                          confidence=0.9999,
                                                          # flags=cv2.SOLVEPNP_EPNP
                                                          )

        if success:
            mkpq_ref = []
            mp3d_ref = []
            for i in range(inliers.shape[0]):
                mkpq_ref.append(mkpq[inliers[i]])
                mp3d_ref.append(mp3d[inliers[i]])
            mp3d_ref = np.array(mp3d_ref, np.float).reshape(-1, 3)
            mkpq_ref = np.array(mkpq_ref, np.float).reshape(-1, 2)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

            rvec, tvec = cv2.solvePnPRefineLM(objectPoints=mp3d_ref,
                                              imagePoints=mkpq_ref,
                                              cameraMatrix=K,
                                              distCoeffs=D,
                                              rvec=rvec,
                                              tvec=tvec,
                                              criteria=criteria,
                                              )
            print('success: ', success, inliers.shape, mp3d.shape, mkpq.shape)
            rvec = np.array(rvec, np.float).reshape(3, 1)
            tvec = np.array([tvec[0, 0], tvec[1, 0], tvec[2, 0]], np.float)
            matches = np.zeros(shape=(mp3d.shape[0], 1), dtype=np.bool)
            for idx in range(inliers.shape[0]):
                matches[inliers[idx]] = 1
            # inliers = inliers.astype(np.bool)
            # success = cv2.solvePnP(mp3d, mkpq, cameraMatrix=K, distCoeffs=D, rvec=rvec, tvec=tvec, flags=cv2.SOLVEPNP_EPNP)
            # criteria = cv2.TermCriteria(cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 20, c)
            # cv2.solvePnPRefineLM()
            # success = cv2.solvePnPRefineLM(mp3d, mkpq, cameraMatrix=K, distCoeffs=D, rvec=rvec, tvec=tvec)
            # print('rvec: ', rvec)
            rot, _ = cv2.Rodrigues(rvec)
            # print("rot: ", rot)
            rot = sciR.from_dcm(rot)
            # print("sci_rot: ", rot)
            qvec = rot.as_quat()
            qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]], np.float)

            # inlier_idxes = [valid[i] for i in range(inliers.shape[0]) if inliers[i]]

            print("qnqme: {:s} dbname: {:s} inliers: {:d}".format(qname, db_name, len(inliers)))
            print("qvec & tvec", qvec, tvec)

            img_match = plot_matches(img1=q_img, img2=db_img,
                                     pts1=mkpq, pts2=mkpdb, inliers=matches)

            cv2.namedWindow("match", cv2.WINDOW_NORMAL)
            cv2.imshow("match", img_match)
            cv2.waitKey(5)

            return qvec, tvec, inliers, inliers.shape[0]


        else:
            print("qnqme: {:s} dbname: {:s} failed".format(qname, db_name))

    closest = db_images[db_ids[0]]
    return closest.qvec, closest.tvec, [], 0


def estimate_pose_with_matcher(matcher, image_dir, reference_sfm, queries, retrieval, features, results, log_fn,
                               ransac_thresh=12, with_label=False, query_list=None, do_covisible_opt=False,
                               inlier_th=10,
                               covisibility_frame=50,
                               vis_dir=None):
    assert reference_sfm.exists(), reference_sfm
    assert retrieval.exists(), retrieval
    assert features.exists(), features

    if vis_dir is not None:
        Path(vis_dir).mkdir(exist_ok=True)
    # assert matches.exists(), matches

    print("Reading queries...")
    queries = parse_image_lists_with_intrinsics(queries)

    print("Reading retrieval...")
    retrieval_dict = parse_retrieval(retrieval)

    print('Reading 3D model from: ', reference_sfm)
    _, db_images, points3D = read_model(str(reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    feature_file = h5py.File(features, 'r')

    poses = {}
    logging.info('Starting localization...')

    failed_cases = []
    all_logs = []

    n_total = 0
    n_failed = 0
    for qname, qinfo in tqdm(queries):
        if query_list is not None:
            if str(qname) not in query_list:
                continue

        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logging.warning(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(db_name_to_id[n])

        # qvec, tvec, inlier_idxs, n_inliers = pose_from_pnp(qname, qinfo, db_ids, db_images, points3D,
        #                                                    feature_file, match_file, thresh=ransac_thresh,
        #                                                    image_dir=image_dir)
        qvec, tvec, n_inliers, logs = pose_from_single_with_matcher(qname, qinfo, db_ids, db_images,
                                                                    points3D,
                                                                    feature_file,
                                                                    thresh=ransac_thresh,
                                                                    image_dir=image_dir,
                                                                    matcher=matcher,
                                                                    with_label=with_label,
                                                                    do_covisility_opt=do_covisible_opt,
                                                                    vis_dir=vis_dir,
                                                                    inlier_th=inlier_th,
                                                                    covisibility_frame=covisibility_frame,
                                                                    )

        all_logs.append(logs)

        n_total += 1

        if n_inliers == 0:
            failed_cases.append(qname)
            n_failed += 1

        poses[qname] = (qvec, tvec)

        print("All {:d}/{:d} failed cases".format(n_failed, n_total))

    logs_path = f'{results}.failed'
    # logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'w') as f:
        for v in failed_cases:
            print(v)
            f.write(v + "\n")

    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            f.write(f'{name} {qvec} {tvec}\n')

    # logs_path = f'{results}_logs.pkl'
    # logging.info(f'Writing logs to {logs_path}...')
    # with open(logs_path, 'wb') as f:
    #     pickle.dump(logs, f)
    logging.info('Done!')

    with open(log_fn, 'w') as f:
        for v in all_logs:
            f.write('{:s} {:s} {:d} {:d}\n'.format(v['qname'], v['dbname'], v['num_inliers'], v['order']))


def main(image_dir, reference_sfm, queries, retrieval, features, matches, results,
         ransac_thresh=12, covisibility_clustering=False):
    assert reference_sfm.exists(), reference_sfm
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists_with_intrinsics(queries)
    retrieval_dict = parse_retrieval(retrieval)

    logging.info('Reading 3D model...')
    _, db_images, points3D = read_model(str(reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logging.info('Starting localization...')

    failed_cases = []
    for qname, qinfo in tqdm(queries):
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logging.warning(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(db_name_to_id[n])

        # qvec, tvec, inlier_idxs, n_inliers = pose_from_pnp(qname, qinfo, db_ids, db_images, points3D,
        #                                                    feature_file, match_file, thresh=ransac_thresh,
        #                                                    image_dir=image_dir)
        qvec, tvec, inlier_idxs, n_inliers = pose_from_single(qname, qinfo, db_ids, db_images, points3D,
                                                              feature_file, match_file, thresh=ransac_thresh,
                                                              image_dir=image_dir)

        if n_inliers == 0:
            failed_cases.append(qname)

        logs[qname] = {
            'db': db_ids,
            'inlier_idxs': inlier_idxs,
            'n_inliers': n_inliers
        }

        poses[qname] = (qvec, tvec)

    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logging.info('Done!')

    print("Failed cases: ", failed_cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default="/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright")
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=str, required=True)
    parser.add_argument('--log_fn', type=str, required=True)
    parser.add_argument('--vis_dir', type=str, default=None)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--inlier_th', type=float, default=10.0)
    parser.add_argument('--covisibility_frame', type=int, default=50)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--with_match', action='store_true')
    parser.add_argument('--with_label', action='store_true')
    parser.add_argument('--do_covisible_opt', action='store_true')
    parser.add_argument('--matcher_method', type=str, default="NNM")
    args = parser.parse_args()

    if args.with_match:
        matcher = Matcher(conf=confs[args.matcher_method])
        matcher = matcher.eval().cuda()

        print("matcher: ", matcher)

        query_list = []
        with open(args.retrieval, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split(' ')
                if l[0] not in query_list:
                    query_list.append(l[0])
        # query_list = []
        # # query_list_fn = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_22_12_03_06_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_ceohem_adam_seg_cls_aug_stylized/loc_by_seg/r2d2-rmax1600-10k-aachen_loc_by_seg_451_feat_max_l10_top5_p3_r3_sinlge_v3_th10_order_more_than_1.txt"
        # query_list_fn = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_23_18_10_31_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_E120_ceohem_adam_seg_cls_aug_stylized/loc_by_seg/r2d2-rmax1600-10k--aachen_loc_by_seg_451_feat_max_l10_top5_p3_r3_sinlge_v3_th10.txt.failed"
        # # with open('failed_cases_451_rec_b16.txt', 'r') as f:
        # with open(query_list_fn, 'r') as f:
        #     lines = f.readlines()
        #     for l in lines:
        #         l = l.strip().split(' ')
        #         query_list.append(l[0])

        log_fn = args.log_fn + '_th' + str(args.inlier_th)
        vis_dir = args.vis_dir + '_th' + str(args.inlier_th)
        results = args.results + '_th' + str(args.inlier_th)
        if args.do_covisible_opt:
            log_fn = log_fn + '_opt' + str(args.covisibility_frame)
            vis_dir = vis_dir + '_opt' + str(args.covisibility_frame)
            results = results + '_opt' + str(args.covisibility_frame)

        log_fn = Path(log_fn + '.log')
        results = Path(results + '.txt')
        vis_dir = Path(vis_dir)

        estimate_pose_with_matcher(matcher=matcher, image_dir=args.image_dir,
                                   reference_sfm=args.reference_sfm,
                                   queries=args.queries,
                                   retrieval=args.retrieval,
                                   features=args.features,
                                   results=results,
                                   log_fn=log_fn,
                                   ransac_thresh=8,
                                   with_label=args.with_label,
                                   query_list=query_list,
                                   do_covisible_opt=args.do_covisible_opt,
                                   vis_dir=vis_dir,
                                   inlier_th=args.inlier_th,
                                   covisibility_frame=args.covisibility_frame,
                                   )
    else:
        main(**args.__dict__)

# opencv-contrib-python         3.4.2.16
# opencv-python                 3.4.2.16
