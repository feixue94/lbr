# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> hloc
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   01/09/2021 20:54
=================================================='''
import torch
import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import time
import logging
from pathlib import Path
import h5py
from tools.common import sort_dict_by_value

from localization.utils.read_write_model import read_model
from localization.utils.parsers import (
    parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair)

from localization.tools import read_retrieval_results, compute_pose_error
from tools.seg_tools import label_to_bgr, read_seg_map_without_group
from localization.fine.matcher import Matcher, confs

from localization.coarse.coarselocalization import CoarseLocalization, prediction_to_labels_v2, prediction_to_labels_v3
from localization.fine.localize_cv2 import pose_from_single, pose_from_single_with_matcher, \
    pose_from_cluster_with_matcher


# torch.backends.cudnn.enabled = False


def extract_db_poses(all_images):
    results = {}
    for db_id in tqdm(all_images):
        fn = all_images[db_id].name
        qvec = all_images[db_id].qvec
        tvec = all_images[db_id].tvec

        results[fn] = (qvec, tvec)

    with open('/scratches/flyer_2/fx221/localization/aachen_v1_1/3D-models/db_poses.txt', 'w') as f:
        for fn in results.keys():
            qvec, tvec = results[fn]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = fn.split('/')[-1]
            f.write(f'{name} {qvec} {tvec}\n')


def extract_db_intrinsics(all_images, cameras):
    results = {}
    for db_id in tqdm(all_images):
        fn = all_images[db_id].name

        db_camera = cameras[all_images[db_id].camera_id]
        # db_cfg = {
        #     'model': db_camera[1],
        #     'width': db_camera[2],
        #     'height': db_camera[3],
        #     'params': db_camera[4],
        # }

        results[fn] = db_camera

        if len(results.keys()) >= 10:
            break

    with open('/scratch2/fx221/localization/aachen_v1_1/3D-models/db_intrinsics.txt', 'w') as f:
        for fn in results.keys():
            cfg = results[fn]
            # qvec, tvec = results[fn]
            # qvec = ' '.join(map(str, qvec))
            # tvec = ' '.join(map(str, tvec))
            model = ' '.join(map(str, cfg[1]))
            width = ' '.join(map(str, cfg[2]))
            height = ' '.join(map(str, cfg[3]))
            f = ' '.join(map(str, cfg[4][0]))
            cx = ' '.join(map(str, cfg[4][1]))
            cy = ' '.join(map(str, cfg[4][2]))
            r = ' '.join(map(str, cfg[4][3]))
            name = fn.split('/')[-1]
            text = '{:s} SIMPLE_RADIAL {:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}'.format(name, width, height, f, cx, cy, r)
            # f.write(f'{name} {model} {width} {height} {f} {cx} {cy} {r}\n')
            f.write(text + '\n')


def extract_covis(all_images, points3D, topk=80):
    db_covis = {}
    for db_id in tqdm(all_images.keys(), total=len(all_images.keys())):
        fn = all_images[db_id].name
        observed = all_images[db_id].point3D_ids
        connected_frames = [j for i in observed if i != -1 for j in points3D[i].image_ids]
        connected_frames = np.unique(connected_frames)

        db_obs = {}
        for db_id in connected_frames:
            p3d_ids = all_images[db_id].point3D_ids
            covisible_p3ds = [v for v in observed if v != -1 and v in p3d_ids]

            if len(covisible_p3ds) <= 200:
                continue
            db_obs[db_id] = len(covisible_p3ds)

        sorted_db_obs = sort_dict_by_value(data=db_obs, reverse=True)
        valid_db_fns = []
        for item in sorted_db_obs:
            valid_db_fns.append((all_images[item[0]].name, item[1]))
        #     if len(valid_db_fns) >= topk:
        #         db_covis[fn] = valid_db_fns
        #         break
        db_covis[fn] = valid_db_fns

    # with open('Aachen_v1.1_db-covis-{:d}-cov30.txt'.format(topk), "w") as f:
    with open('Aachen_v1.1_db-covis200.txt', "w") as f:
        for fn in db_covis.keys():
            cans = db_covis[fn]
            for c in cans:
                f.write("{:s} {:s} {:d}\n".format(fn, c[0], c[1]))


def global_search(q_feat, db_feats, topk=50):
    all_db_feat = []
    all_db_fns = []
    for fn in db_feats.keys():
        all_db_fns.append(fn)
        all_db_feat.append(db_feats[fn])

    q_feat_torch = torch.from_numpy(q_feat).cuda()
    if len(q_feat_torch.shape) == 1:
        q_feat_torch = q_feat_torch.unsqueeze(0)

    all_db_feat_torch = torch.from_numpy(np.vstack(all_db_feat)).cuda()
    if len(all_db_feat_torch.shape) == 1:
        all_db_feat_torch = all_db_feat_torch.unsqueeze(0)

    with torch.no_grad():
        sim = q_feat_torch @ all_db_feat_torch.t()
    k = topk
    if k > all_db_feat_torch.shape[0]:
        k = all_db_feat_torch.shape[0]

    scores, idxes = torch.topk(sim, dim=1, largest=True, k=k)

    results = []
    for i in range(k):
        results.append((all_db_fns[idxes[0, i]], scores[0, i].cpu().numpy()))

    return results


def run(args):
    gt_pose_fn = '/scratches/flyer_2/fx221/localization/outputs_hloc/aachen_v1_1/Aachen-v1.1_hloc_superpoint_n4096_r1600+superglue_netvlad50.txt'
    gt_poses = {}
    with open(gt_pose_fn, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            gt_poses[l[0]] = np.array([float(v) for v in l[1:]])

    save_fn = args.retrieval_type
    # save_fn = "test"
    if args.retrieval is not None:
        nv_retrievals = read_retrieval_results(args.retrieval)
        # save_fn = save_fn + "nv"
    else:
        nv_retrievals = {}

    print('nv_retrieval: ', len(nv_retrievals.keys()))


    # test_cat = 'night'
    test_cat = 'day'
    # save_fn = save_fn + 'test'
    save_fn = save_fn
    if test_cat == 'day':
        save_fn = save_fn + 'dt'
    else:
        save_fn = save_fn + 'nt'
    # save_fn = save_fn + 'rbt_time_dt'

    save_root = args.save_root
    show_seg = args.show_seg
    show_seg = False

    feat_type = args.feature_type
    k_seg = args.k_seg
    k_can = args.k_can
    k_rec = args.k_rec
    version = args.version

    matcher_name = args.matcher_method
    local_feat_name = args.features.as_posix().split("/")[-1].split(".")[0]

    save_fn = save_fn + "l{:d}c{:d}r{:d}{:s}{:s}fh{:.3f}{:s}".format(k_seg,
                                                                     k_can,
                                                                     k_rec,
                                                                     local_feat_name,
                                                                     matcher_name,
                                                                     args.global_score_th,
                                                                     args.init_type,
                                                                     )
    if version == 'v6':
        map_gid_rgb = read_seg_map_without_group("datasets/aachen/aachen_grgb_gid_v5.txt")
        weight_name = args.weight_name
        db_imglist_fn = "datasets/aachen/aachen_db_imglist.txt"
        save_tmp_fn = 'aachen_452'

    q_pred_dir = osp.join(save_root, weight_name, "confidence")
    # save_seg_dir = osp.join(save_root, weight_name, "masks")
    seg_dir = osp.join(save_root, weight_name, "masks")
    # db_seg_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/global_seg_instance"
    db_seg_dir = args.seg_dir
    q_imglist_fn = "datasets/aachen/aachen_query_imglist.txt"
    save_dir = osp.join(save_root, weight_name, "loc_by_seg")
    save_fn = osp.join(save_dir, save_fn)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    q_imglist = []
    with open(q_imglist_fn, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()

            q_imglist.append(l)

    Localizer = CoarseLocalization()
    Localizer.load_db_rec(seg_dir=db_seg_dir, list_fn=db_imglist_fn,
                          save_tmp_fn=save_tmp_fn)

    if feat_type.find('netvlad') >= 0 or feat_type.find('gem') >= 0:
        print('Load global feats from {:s}'.format(osp.join(osp.join(save_root, weight_name, feat_type + '.npy'))))
        use_global_feat = True
        global_feats = np.load(osp.join(save_root, weight_name, feat_type + '.npy'), allow_pickle=True).item()
        db_feats = global_feats
    else:
        print('Load global feats from {:s}'.format(q_pred_dir))
        use_global_feat = False
        db_feats = {}

    # preparation for fine localization
    with_label = args.with_label
    if args.with_match:
        matcher = Matcher(conf=confs[args.matcher_method])
        matcher = matcher.eval().cuda()

        print("matcher: ", matcher)

        if args.matcher_method in ['NNM', 'NNMR']:
            with_label = False

    print("queries: ", args.queries)
    queries = parse_image_lists_with_intrinsics(args.queries)
    print('Reading 3D model from: ', args.reference_sfm)
    _, db_images, points3D = read_model(str(args.reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}

    # extract_covis(all_images=db_images, points3D=points3D, topk=80)
    # extract_db_poses(all_images=db_images)
    # exit(0)

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
            args.opt_thresh)) + "r" + str(
            args.radius)

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

    poses = {}
    logging.info('Starting localization...')
    print("save_fn: ", log_fn)
    failed_cases = []
    all_logs = []

    n_total = 0
    n_failed = 0
    n_top1 = 0
    n_rec = 0
    n_nv = 0

    full_log_info = ''

    n_q = 0
    success_ratio = [0, 0, 0, 0]
    for qname, qinfo in tqdm(queries):
        # if qname not in q_fn_lists_error:
        #     continue
        # if qname not in [
        #     "query/night/nexus5x/IMG_20161227_172626.jpg",
        #     "query/night/nexus5x/IMG_20161227_192316.jpg",
        #     "query/night/nexus5x/IMG_20161227_173326.jpg",
        #     "query/night/nexus5x_additional_night/IMG_20170702_005301.jpg",
        # ]:
        #     continue

        # if qname.find('night') >= 0:
        if qname.find(test_cat) < 0:
            continue

        if show_seg:
            q_seg = cv2.imread(osp.join(seg_dir, qname.replace("jpg", "png")))
            # cv2.imshow("q_seg", q_seg)
        else:
            q_seg = None

        pred = np.load(osp.join(q_pred_dir, qname.split(".")[0] + ".npy"), allow_pickle=True).item()
        pred_confs = pred["confidence"]
        pred_labels = pred["ids"]

        if use_global_feat:
            q_feat = global_feats[qname]
        else:
            q_feat = pred[feat_type]

        # q_feat = q_feat - np.mean(q_feat)
        time_start = time.time()
        q_feat = q_feat / np.linalg.norm(q_feat)
        q_f, q_label, q_uids, q_uid_confs = prediction_to_labels_v3(pred_conf=pred_confs, pred_labels=pred_labels,
                                                                    cnt_th=5000,
                                                                    cnt_labels=k_seg,
                                                                    map_gid_rgb=map_gid_rgb)
        # q_seg = q_seg[:, q_seg.shape[1] // 3:q_seg.shape[1] // 3 * 2, ]
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
            cands = Localizer.loc_by_rec_v3_avg(query_seg=q_seg, query_feat=q_feat, db_feats=db_feats,
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
                    global_score[c] = 1.0

                    inlier_ths.append(args.nv_th)
                    retrieval_sources.append("nv")

        elif args.init_type.find('clu') >= 0:
            cands = Localizer.loc_by_rec_v3_avg_cluster(query_seg=q_seg, query_feat=q_feat, db_feats=db_feats,
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
                    global_score[c] = 1.0
                    nv_db_ids.append(db_name_to_id[c])

                # nv_custers = do_covisibility_clustering(frame_ids=nv_db_ids, all_images=db_images, points3D=points3D)
                # for nv_cls in nv_custers:
                for nv_cls in nv_db_ids:
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
                                                                     # seg_dir=seg_dir if show_seg else None,
                                                                     seg_dir=db_seg_dir if show_seg else None,
                                                                     q_seg=q_seg,
                                                                     log_info='',
                                                                     opt_type=args.opt_type,
                                                                     plus05=args.plus05,
                                                                     do_cluster_check=args.do_cluster_check,
                                                                     iters=args.iters,
                                                                     radius=args.radius,
                                                                     obs_th=args.obs_thresh,
                                                                     opt_th=args.opt_thresh,
                                                                     with_dist=args.with_dist,
                                                                     inlier_ths=inlier_ths,
                                                                     retrieval_sources=retrieval_sources,
                                                                     gt_qvec=gt_poses[qname.split('/')[-1]][:4],
                                                                     gt_tvec=gt_poses[qname.split('/')[-1]][4:],
                                                                     # show_match=False,
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
        pose_error = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec,
                                        gt_qcw=gt_poses[qname.split('/')[-1]][:4],
                                        gt_tcw=gt_poses[qname.split('/')[-1]][4:])

        if pose_error[0] <= 2 and pose_error[1] <= 0.25:
            success_ratio[0] += 1
            success_ratio[1] += 1
            success_ratio[2] += 1

        elif pose_error[0] <= 5 and pose_error[1] <= 0.5:
            success_ratio[1] += 1
            success_ratio[2] += 1
        elif pose_error[0] <= 10 and pose_error[1] <= 5:
            success_ratio[2] += 1
        success_ratio[3] += 1

        print_text = "All {:d}/{:d} failed cases top@1:{:.2f}, rec:{:.2f} nv:{:.2f}, {:d}, time[cs/fn]: {:.2f}/{:.2f}, " \
                     "error:{:.2f}d/{:.2f}m, {:.1f}/{:.1f}/{:.1f}".format(
            n_failed, n_total,
            n_top1 / n_total,
            n_rec / n_total,
            n_nv / n_total,
            len(db_feats.keys()),
            time_coarse - time_start,
            time_full - time_coarse,
            pose_error[0],
            pose_error[1],
            success_ratio[0] / success_ratio[3] * 100.,
            success_ratio[1] / success_ratio[3] * 100.,
            success_ratio[2] / success_ratio[3] * 100.,
        )
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


def run_robotcar(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default="/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright", )
    parser.add_argument('--seg_dir', type=str,
                        default="/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright", )
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--rec_th', type=int, default=50)
    parser.add_argument('--nv_th', type=int, default=20)
    parser.add_argument('--covisibility_frame', type=int, default=50)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--with_match', action='store_true')
    parser.add_argument('--with_label', action='store_true')
    parser.add_argument('--do_cluster_check', action='store_true')
    parser.add_argument('--do_covisible_opt', action='store_true')
    parser.add_argument('--opt_type', type=str, default='cluster')
    parser.add_argument('--show_seg', action='store_true', default=False)
    parser.add_argument('--plus05', action='store_true', default=False)
    parser.add_argument('--matcher_method', type=str, default="NNM")

    parser.add_argument('--obs_thresh', type=float, default=50.0)
    parser.add_argument('--opt_thresh', type=float, default=12.0)
    parser.add_argument('--with_dist', action='store_true', default=False)

    parser.add_argument('--depth_th', type=float, default=0)
    parser.add_argument('--radius', type=int, default=15)
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

    if args.dataset == "aachen":
        run(args=args)
    elif args.dataset == "robotcar":
        run_robotcar(args=args)
