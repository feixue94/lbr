# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   lbr -> localizer
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   06/04/2022 15:07
=================================================='''
import torch
import os
import os.path as osp
from tqdm import tqdm
import argparse
import time
import logging
import h5py
import numpy as np
from pathlib import Path

from localization.utils.read_write_model import read_model
from localization.utils.parsers import (
    parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair)

from tools.loc_tools import read_retrieval_results, compute_pose_error
from tools.seg_tools import label_to_bgr, read_seg_map_without_group
from localization.fine.matcher import Matcher, confs
from localization.coarse.coarselocalization import CoarseLocalization, prediction_to_labels_v3
from localization.fine.localize_cv2 import pose_from_cluster_with_matcher


def run(args):
    # for visualization only (not used for localization)
    if args.gt_pose_fn is not None:
        gt_poses = {}
        with open(args.gt_pose_fn, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split(' ')
                gt_poses[l[0]] = {
                    'qvec': np.array([float(v) for v in l[1:5]], float),
                    'tvec': np.array([float(v) for v in l[5:]], float),
                }
                # print(l[0])
                # exit(0)
    else:
        gt_poses = {}

    retrieval_type = args.retrieval_type
    save_fn = retrieval_type
    if args.retrieval is not None:
        nv_retrievals = read_retrieval_results(args.retrieval)
    else:
        nv_retrievals = {}

    save_root = args.save_root  # path to save
    feat_type = args.feature_type  # feature used for local reference search
    k_seg = args.k_seg
    k_can = args.k_can
    k_rec = args.k_rec
    matcher_name = args.matcher_method  # matching method
    local_feat_name = args.features.as_posix().split("/")[-1].split(".")[0]  # name of local features

    save_fn = save_fn + "l{:d}c{:d}r{:d}{:s}{:s}fh{:.3f}{:s}".format(k_seg,
                                                                     k_can,
                                                                     k_rec,
                                                                     local_feat_name,
                                                                     matcher_name,
                                                                     args.global_score_th,
                                                                     args.init_type,
                                                                     )
    if retrieval_type.find("lr") >= 0:
        weight_name = args.weight_name
        map_gid_rgb = read_seg_map_without_group(args.map_gid_rgb_fn)
        db_imglist_fn = args.db_imglist_fn
        db_instance_fn = args.db_instance_fn

        q_pred_dir = osp.join(save_root, weight_name, "confidence")
        pred_seg_dir = osp.join(save_root, weight_name, "masks")
        db_seg_dir = args.seg_dir
        save_dir = osp.join(save_root, weight_name, "loc_by_seg")
    else:
        save_dir = osp.join(save_root, retrieval_type)

    # db_seg_dir = args.seg_dir
    save_dir = osp.join(save_dir, weight_name, "loc_by_seg")
    save_fn = osp.join(save_dir, save_fn)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if retrieval_type.find("lr") >= 0:
        CLocalizer = CoarseLocalization()
        CLocalizer.load_db_rec(seg_dir=db_seg_dir, list_fn=db_imglist_fn,
                               save_tmp_fn=db_instance_fn)

    # preparation for fine localization
    with_label = args.with_label
    if args.with_match:
        matcher = Matcher(conf=confs[args.matcher_method])
        matcher = matcher.eval().cuda()
        if args.matcher_method in ['NNM', 'NNMR']:
            with_label = False

    queries = parse_image_lists_with_intrinsics(args.queries)
    _, db_images, points3D = read_model(str(args.reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    feature_file = h5py.File(args.features, 'r')

    tag = 'ir' + str(int(args.rec_th))
    if args.retrieval is not None:
        tag = tag + "iv" + str(int(args.nv_th))
    tag = tag + 'rs' + str(int(args.ransac_thresh))
    log_fn = save_fn + tag
    vis_dir = save_fn + tag
    results = save_fn + tag

    if args.do_covisible_opt:
        tag = "o" + str(int(args.obs_thresh)) + 'op' + str(int(args.covisibility_frame))
        tag = tag + args.opt_type + "th" + str(int(
            args.opt_thresh)) + "r" + str(args.radius)
        if args.iters > 0:
            tag = tag + "i" + str(int(args.iters))
        log_fn = log_fn + tag
        vis_dir = vis_dir + tag
        results = results + tag
    full_log_fn = log_fn + '_full.log'
    log_fn = Path(log_fn + '.log')
    results = Path(results + '.txt')
    vis_dir = Path(vis_dir)
    if vis_dir is not None:
        Path(vis_dir).mkdir(exist_ok=True)
    print("save_fn: ", log_fn)

    logging.info('Starting localization...')
    db_feats = {}
    poses = {}
    failed_cases = []
    all_logs = []
    n_total = 0
    n_failed = 0
    n_top1 = 0
    n_rec = 0
    n_nv = 0
    full_log_info = ''

    error_ths = ((0.25, 2), (0.5, 5), (5, 10))
    success = [0, 0, 0]
    n_gt_total = 0
    for qname, qinfo in tqdm(queries):
        # if qname.find('night') >= 0:
        # if qname.find(test_cat) < 0:
        #     continue
        time_start = time.time()
        if retrieval_type.find("lr") >= 0:
            pred = np.load(osp.join(q_pred_dir, qname.split(".")[0] + ".npy"), allow_pickle=True).item()
            pred_confs = pred["confidence"]
            pred_labels = pred["ids"]
            q_feat = pred[feat_type]

            q_feat = q_feat / np.linalg.norm(q_feat)
            q_f, q_label, q_uids, q_uid_confs = prediction_to_labels_v3(pred_conf=pred_confs,
                                                                        pred_labels=pred_labels,
                                                                        cnt_th=5000,
                                                                        cnt_labels=k_seg,
                                                                        map_gid_rgb=map_gid_rgb)
            q_seg = label_to_bgr(label=q_label, maps=map_gid_rgb)
            q_rgb_confs = {}
            for v in q_uid_confs.keys():
                q_rgb_confs[map_gid_rgb[v]] = q_uid_confs[v]

        db_ids = []
        global_score = {}
        n_cans_total = 200
        n_cans = 0
        inlier_ths = []
        retrieval_sources = []

        if args.init_type.find('sing') >= 0:
            if retrieval_type.find("lr") >= 0:
                cands = CLocalizer.loc_by_rec_v3_avg(query_seg=q_seg, query_feat=q_feat, db_feats=db_feats,
                                                     db_feat_dir=q_pred_dir,
                                                     k=k_can,
                                                     q_uids_confs=q_rgb_confs,
                                                     feat_type=feat_type,
                                                     log_info=full_log_info,
                                                     k_rec=k_rec,
                                                     )

                for idx_c, c_item in enumerate(cands):
                    if c_item[0] not in db_name_to_id:
                        logging.warning(f'Image {c_item[0]} was retrieved but not in database')
                        continue
                    if c_item[1] < args.global_score_th:
                        continue

                    full_log_info += ('{:s} {:s} {:.5f} by recognition\n'.format(qname, c_item[0], c_item[1]))
                    db_ids.append([db_name_to_id[c_item[0]]])
                    global_score[c_item[0]] = c_item[1]

                    inlier_ths.append(args.rec_th)
                    retrieval_sources.append("rec")

                    n_cans += 1
                    if n_cans >= n_cans_total:
                        break

            if qname in nv_retrievals.keys():
                nv_cans = nv_retrievals[qname]
                for c in nv_cans:
                    if c not in db_name_to_id:
                        logging.warning(f'Image {c} was retrieved but not in database')
                        continue
                    full_log_info += ('{:s} {:s} {:.2f} by global search\n'.format(qname, c, 1))
                    db_ids.append([db_name_to_id[c]])

                    if c not in global_score.keys():
                        global_score[c] = 1.0

                    inlier_ths.append(args.nv_th)
                    retrieval_sources.append("nv")

        elif args.init_type.find('clu') >= 0:
            if retrieval_type.find("lr") >= 0:
                cands = CLocalizer.loc_by_rec_v3_avg_cluster(query_seg=q_seg, query_feat=q_feat, db_feats=db_feats,
                                                             db_feat_dir=q_pred_dir,
                                                             k=k_can,
                                                             q_uids_confs=q_rgb_confs,
                                                             feat_type=feat_type,
                                                             log_info=full_log_info,
                                                             k_rec=k_rec,
                                                             )
                # cads = [[(c0, s0), (c1, s1)] [...], [...]]
                for cls_i in cands:
                    if len(cls_i) == 0:
                        continue
                    cls_cans = []
                    for idx_c, c_item in enumerate(cls_i):
                        if c_item[0] not in db_name_to_id:
                            logging.warning(f'Image {c_item[0]} was retrieved but not in database')
                            continue
                        if c_item[1] < args.global_score_th:
                            continue

                        if db_name_to_id[c_item[0]] in cls_cans:
                            continue

                        full_log_info += ('{:s} {:s} {:.5f} by recognition\n'.format(qname, c_item[0], c_item[1]))
                        global_score[c_item[0]] = c_item[1]

                        cls_cans.append(db_name_to_id[c_item[0]])
                        n_cans += 1
                    if len(cls_cans) == 0:
                        continue
                    db_ids.append(cls_cans)
                    inlier_ths.append(args.rec_th)
                    retrieval_sources.append("rec")

                    if n_cans >= n_cans_total:
                        break

            if qname in nv_retrievals.keys():
                nv_db_ids = []
                nv_cans = nv_retrievals[qname]
                for c in nv_cans:
                    if c not in db_name_to_id:
                        logging.warning(f'Image {c} was retrieved but not in database')
                        continue

                    full_log_info += ('{:s} {:s} {:.2f} by global search\n'.format(qname, c, 1))

                    if c not in global_score.keys():
                        global_score[c] = 1.0
                    nv_db_ids.append(db_name_to_id[c])

                for rid, nv_cls in enumerate(nv_db_ids):
                    db_ids.append([nv_cls])
                    inlier_ths.append(args.nv_th)
                    retrieval_sources.append("nv")

        time_coarse = time.time()
        qvec, tvec, n_inliers, logs = pose_from_cluster_with_matcher(qname, qinfo, db_ids, db_images,
                                                                     points3D,
                                                                     feature_file,
                                                                     thresh=args.ransac_thresh,
                                                                     image_dir=args.image_dir,
                                                                     matcher=matcher,
                                                                     with_label=with_label,
                                                                     do_covisility_opt=args.do_covisible_opt,
                                                                     vis_dir=vis_dir,
                                                                     inlier_th=args.rec_th,
                                                                     covisibility_frame=args.covisibility_frame,
                                                                     global_score=global_score,
                                                                     seg_dir=None,
                                                                     q_seg=q_seg,
                                                                     log_info='',
                                                                     opt_type=args.opt_type,
                                                                     iters=args.iters,
                                                                     radius=args.radius,
                                                                     obs_th=args.obs_thresh,
                                                                     opt_th=args.opt_thresh,
                                                                     inlier_ths=inlier_ths,
                                                                     retrieval_sources=retrieval_sources,
                                                                     gt_qvec=gt_poses[qname.split('/')[-1]][
                                                                         'qvec'] if qname.split('/')[
                                                                                        -1] in gt_poses.keys() else None,
                                                                     gt_tvec=gt_poses[qname.split('/')[-1]][
                                                                         'tvec'] if qname.split('/')[
                                                                                        -1] in gt_poses.keys() else None,
                                                                     )

        time_full = time.time()
        all_logs.append(logs)
        n_total += 1
        if n_inliers == 0:
            failed_cases.append(qname)
            n_failed += 1
        if logs['order'] == 1:
            n_top1 += 1

        if logs['ret_source'] == "rec":
            n_rec += 1
        elif logs['ret_source'] == "nv":
            n_nv += 1
        full_log_info = full_log_info + logs['log_info']

        poses[qname] = (qvec, tvec)

        print_text = "All {:d}/{:d} failed cases top@1:{:.2f}, rec:{:.2f} nv:{:.2f}, {:d}, time[cs/fn]: {:.2f}/{:.2f}".format(
            n_failed, n_total,
            n_top1 / n_total,
            n_rec / n_total,
            n_nv / n_total,
            len(db_feats.keys()),
            time_coarse - time_start,
            time_full - time_coarse,
        )

        if qname.split('/')[-1] in gt_poses.keys():
            gt_qvec = gt_poses[qname.split('/')[-1]]['qvec']
            gt_tvec = gt_poses[qname.split('/')[-1]]['tvec']

            q_error, t_error, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=gt_qvec, gt_tcw=gt_tvec)

            for error_idx, th in enumerate(error_ths):
                if t_error <= th[0] and q_error <= th[1]:
                    success[error_idx] += 1
            n_gt_total += 1
            print_text += (
                ', q_error:{:.2f} t_error:{:.2f} {:d}/{:d}/{:d}/{:d}'.format(q_error, t_error, success[0], success[1],
                                                                            success[2], n_gt_total))

        print(print_text)
        full_log_info += (print_text + "\n")

    logs_path = f'{results}.failed'
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

    with open(log_fn, 'w') as f:
        for v in all_logs:
            f.write('{:s} {:s} {:d} {:d}\n'.format(v['qname'], v['dbname'], v['num_inliers'], v['order']))

    with open(full_log_fn, 'w') as f:
        f.write(full_log_info)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default="/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright", )
    parser.add_argument('--seg_dir', type=str,
                        default="/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright", )
    parser.add_argument('--map_gid_rgb_fn', type=str, required=True)
    parser.add_argument('--db_imglist_fn', type=str, required=True)
    parser.add_argument('--db_instance_fn', type=str, required=True)
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--rec_th', type=int, default=50)
    parser.add_argument('--nv_th', type=int, default=20)
    parser.add_argument('--covisibility_frame', type=int, default=50)
    parser.add_argument('--with_match', action='store_true')
    parser.add_argument('--with_label', action='store_true')
    parser.add_argument('--do_covisible_opt', action='store_true')
    parser.add_argument('--opt_type', type=str, default='cluster')
    parser.add_argument('--show_seg', action='store_true', default=False)
    parser.add_argument('--matcher_method', type=str, default="NNM")

    parser.add_argument('--obs_thresh', type=float, default=3)
    parser.add_argument('--opt_thresh', type=float, default=12.0)
    parser.add_argument('--radius', type=int, default=20)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--k_seg', type=int, default=10)
    parser.add_argument('--k_can', type=int, default=5)
    parser.add_argument('--k_rec', type=int, default=30)
    parser.add_argument('--feature_type', type=str, default="feat_max")
    parser.add_argument('--init_type', type=str, default="single")
    parser.add_argument('--global_score_th', type=float, default=-1.)
    parser.add_argument('--save_root', type=str, default="/data/cornucopia/fx221/exp/shloc/aachen")
    parser.add_argument('--weight_name', type=str, required=True)
    parser.add_argument('--retrieval', type=Path, default=None)
    parser.add_argument('--retrieval_type', type=str, default="lrnv")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--gt_pose_fn', type=str, default=None)
    parser.add_argument('--only_gt', type=int, default=0)

    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    run(args=args)
