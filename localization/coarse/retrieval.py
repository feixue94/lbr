# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> retrieval
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/09/2021 10:25
=================================================='''
import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
import cv2
from localization.utils.read_write_model import read_model



def extract_db_poses(all_images, save_fn):
    results = {}
    for db_id in tqdm(all_images):
        fn = all_images[db_id].name
        print(fn)
        qvec = all_images[db_id].qvec
        tvec = all_images[db_id].tvec

        results[fn] = (qvec, tvec)

    # with open('/scratches/flyer_2/fx221/localization/aachen_v1_1/3D-models/db_poses.txt', 'w') as f:
    with open(save_fn, 'w') as f:
        for fn in results.keys():
            qvec, tvec = results[fn]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            # name = fn.split('/')[-1]
            name = fn
            f.write(f'{name} {qvec} {tvec}\n')


def patch_netvlad():
    feat_path = osp.join(root_dir, 'globalfeats.npy')
    all_feats = np.load(feat_path, allow_pickle=True).item()
    k = 50

    query_fns = []
    query_feats = []

    db_fns = []
    db_feats = []
    for fn in all_feats.keys():
        if fn.find('query') >= 0:
            query_fns.append(fn)
            query_feats.append(all_feats[fn])
        else:
            db_fns.append(fn)
            db_feats.append(all_feats[fn])

    print('Load {:d} query & {:d} db images'.format(len(query_fns), len(db_fns)))
    db_feats = np.vstack(db_feats)
    db_feats_torch = torch.from_numpy(db_feats).cuda()
    # print(db_feats_torch.shape)

    results = {}
    for q_fn, q_feat in tqdm(zip(query_fns, query_feats), total=len(query_fns)):
        q_feat_torch = torch.from_numpy(q_feat).cuda()
        # print(q_feat_torch.shape)
        # exit(0)

        if q_fn not in ["query/night/nexus5x_additional_night/IMG_20170702_005301.jpg"]:
            continue

        dist = q_feat_torch @ db_feats_torch.t()
        pred_dist, pred_idx = torch.topk(dist, k=50, largest=True)

        cans = []
        for i in range(pred_idx.shape[1]):
            cans.append(db_fns[pred_idx[0, i]])

        results[q_fn] = cans
        q_img = cv2.imread(osp.join(image_dir, q_fn))
        q_img = cv2.resize(q_img, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow('q', q_img)
        for c in cans:
            c_img = cv2.imread(osp.join(image_dir, c))
            c_img = cv2.resize(c_img, dsize=None, fx=0.5, fy=0.5)
            cv2.imshow('c', c_img)
            cv2.waitKey(0)

    # with open(osp.join(root_dir, 'patch-pitts-netvlad{:d}.txt'.format(k)), 'w') as f:
    #     for fn in results.keys():
    #         cans = results[fn]
    #         for c in cans:
    #             text = '{:s} {:s}'.format(fn, c)
    #             f.write(text + '\n')


def patch_netvlad_robotcar():
    image_dir = "/scratch2/fx221/localization/RobotCar-Seasons/images"
    root_dir = "/scratch2/fx221/exp/retrieval/patch-netvlad-480-640/robotcar"
    feat_path = osp.join(root_dir, 'globalfeats.npy')
    all_feats = np.load(feat_path, allow_pickle=True).item()
    k = 50

    query_fns = []
    query_feats = []

    db_fns = []
    db_feats = []
    for fn in all_feats.keys():
        if fn.find('reference') >= 0:
            if fn.find("rear") >= 0:
                db_fns.append(fn)
                db_feats.append(all_feats[fn])
        else:
            query_fns.append(fn)
            query_feats.append(all_feats[fn])

    print('Load {:d} query & {:d} db images'.format(len(query_fns), len(db_fns)))
    db_feats = np.vstack(db_feats)
    db_feats_torch = torch.from_numpy(db_feats).cuda()
    # print(db_feats_torch.shape)

    results = {}
    for q_fn, q_feat in tqdm(zip(query_fns, query_feats), total=len(query_fns)):
        q_feat_torch = torch.from_numpy(q_feat).cuda()
        # print(q_feat_torch.shape)
        # exit(0)

        # if q_fn not in ["query/night/nexus5x_additional_night/IMG_20170702_005301.jpg"]:
        #     continue

        dist = q_feat_torch @ db_feats_torch.t()
        pred_dist, pred_idx = torch.topk(dist, k=50, largest=True)

        cans = []
        for i in range(pred_idx.shape[1]):
            cans.append(db_fns[pred_idx[0, i]])

        results[q_fn] = cans
        # q_img = cv2.imread(osp.join(image_dir, q_fn))
        # q_img = cv2.resize(q_img, dsize=None, fx=0.5, fy=0.5)
        # cv2.imshow('q', q_img)
        # for c in cans:
        #     c_img = cv2.imread(osp.join(image_dir, c))
        #     c_img = cv2.resize(c_img, dsize=None, fx=0.5, fy=0.5)
        #     cv2.imshow('c', c_img)
        #     cv2.waitKey(0)

    with open(osp.join(root_dir, 'robotcar_pairs-patch-netvlad{:d}.txt'.format(k)), 'w') as f:
        for fn in results.keys():
            cans = results[fn]
            for c in cans:
                text = '{:s} {:s}'.format(fn, c)
                f.write(text + '\n')


def pose_from_retrieval_robotcar():
    db_pose_robotcar_fn = '/scratch2/fx221/localization/RobotCar-Seasons/3D-models/db_poses.txt'

    db_pose_fn = db_pose_robotcar_fn

    # query_path_aachenv11 = '/scratch2/fx221/localization/aachen_v1_1/queries/day_night_time_queries_with_intrinsics.txt'
    query_path_robotcar = '/scratch2/fx221/localization/RobotCar-Seasons/queries_with_intrinsics_rear.txt'

    query_fns = []
    with open(query_path_robotcar, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(' ')[0]
            query_fns.append(l)
    # print(query_fns)
    # exit(0)
    root_dir = '/scratch2/fx221/exp/retrieval/patch-netvlad-480-640/robotcar'
    # retrieval_fn = osp.join(root_dir, 'pairs-patch-pitts-netvlad50.txt')
    # retrieval_fn = osp.join(root_dir, 'pairs-patch-pitts-netvlad50.txt')
    # save_fn = osp.join(root_dir, 'netvladv1.1_patch-pitts-netvlad50.txt')
    # save_fn = osp.join(root_dir, 'robotcar_patch-pitts-netvlad50.txt')
    retrieval_fn = osp.join(root_dir, 'robotcar_pairs-patch-netvlad50.txt')
    save_fn = osp.join(root_dir, 'robotcar_patch-netvlad50-percam-perloc.txt')



    # root_dir = '/scratch2/fx221/exp/retrieval/dir-1024'
    # retrieval_fn = osp.join(root_dir, 'robotcar-pairs-dirnet20-percam-perloc.txt')
    # save_fn = osp.join(root_dir, 'netvladv1.1_dir50.txt')
    # save_fn = osp.join(root_dir, 'robotcar-dir20-percam-perloc.txt')

    # root_dir = '/scratch2/fx221/exp/retrieval/netvlad-1024'
    # retrieval_fn = osp.join(root_dir, 'pairs-query-netvlad50.txt')
    # save_fn = osp.join(root_dir, 'aachenv1.1_netvlad50.txt')

    # retrieval_fn = osp.join(root_dir, 'robotcar-pairs-netvlad20-percam-perloc.txt')
    # save_fn = osp.join(root_dir, 'robotcar-netvlad20-percam-perloc.txt')

    # root_dir = '/scratches/flyer_2/fx221/exp/retrieval/patch-netvlad-480-640'

    db_poses = {}
    with open(db_pose_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            db_poses[l[0]] = l[1:]

    retrieval_poses = {}

    with open(retrieval_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(' ')
            q_fn = l[0]
            c_fn = l[1]

            # print(q_fn)

            if q_fn not in query_fns:
                continue

            if q_fn in retrieval_poses.keys():
                continue

            if c_fn not in db_poses.keys():
                continue

            retrieval_poses[q_fn] = db_poses[c_fn]

    with open(save_fn, 'w') as f:
        for fn in retrieval_poses.keys():
            text = fn.split('/')[-1]
            camera = fn.split('/')[1]
            text = '{:s}/{:s}'.format(camera, text)
            for v in retrieval_poses[fn]:
                text = text + ' ' + v
            f.write(text + '\n')



def pose_from_retrieval():
    db_pose_aachenv11_fn = '/scratch2/fx221/localization/aachen_v1_1/3D-models/db_poses.txt'
    db_pose_robotcar_fn = '/scratch2/fx221/localization/RobotCar-Seasons/3D-models/db_poses.txt'

    db_pose_fn = db_pose_robotcar_fn

    # query_path_aachenv11 = '/scratch2/fx221/localization/aachen_v1_1/queries/day_night_time_queries_with_intrinsics.txt'
    query_path_robotcar = '/scratch2/fx221/localization/RobotCar-Seasons/queries_with_intrinsics_rear.txt'

    query_fns = []
    with open(query_path_robotcar, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(' ')[0]
            query_fns.append(l)
    # print(query_fns)
    # exit(0)
    # root_dir = '/scratch2/fx221/exp/retrieval/patch-netvlad-480-640'
    # retrieval_fn = osp.join(root_dir, 'pairs-patch-pitts-netvlad50.txt')
    # retrieval_fn = osp.join(root_dir, 'pairs-patch-pitts-netvlad50.txt')
    # save_fn = osp.join(root_dir, 'netvladv1.1_patch-pitts-netvlad50.txt')
    # save_fn = osp.join(root_dir, 'robotcar_patch-pitts-netvlad50.txt')

    # root_dir = '/scratch2/fx221/exp/retrieval/dir-1024'
    # retrieval_fn = osp.join(root_dir, 'robotcar-pairs-dirnet20-percam-perloc.txt')
    # save_fn = osp.join(root_dir, 'netvladv1.1_dir50.txt')
    # save_fn = osp.join(root_dir, 'robotcar-dir20-percam-perloc.txt')

    root_dir = '/scratch2/fx221/exp/retrieval/netvlad-1024'
    # retrieval_fn = osp.join(root_dir, 'pairs-query-netvlad50.txt')
    # save_fn = osp.join(root_dir, 'aachenv1.1_netvlad50.txt')

    retrieval_fn = osp.join(root_dir, 'robotcar-pairs-netvlad20-percam-perloc.txt')
    save_fn = osp.join(root_dir, 'robotcar-netvlad20-percam-perloc.txt')

    # root_dir = '/scratches/flyer_2/fx221/exp/retrieval/patch-netvlad-480-640'
    image_dir = '/scratches/flyer_2/fx221/localization/aachen_v1_1/images/images_upright'

    db_poses = {}
    with open(db_pose_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            db_poses[l[0]] = l[1:]

    retrieval_poses = {}

    with open(retrieval_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(' ')
            q_fn = l[0]
            c_fn = l[1]

            # print(q_fn)

            if q_fn not in query_fns:
                continue

            if q_fn in retrieval_poses.keys():
                continue

            c_fn = c_fn.split('/')[-1]
            if c_fn not in db_poses.keys():
                continue

            retrieval_poses[q_fn] = db_poses[c_fn]

    with open(save_fn, 'w') as f:
        for fn in retrieval_poses.keys():
            text = fn.split('/')[-1]
            for v in retrieval_poses[fn]:
                text = text + ' ' + v
            f.write(text + '\n')


def read_fns(path):
    output = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(' ')
            output.append(l[0])
    return output

if __name__ == '__main__':
    pose_from_retrieval_robotcar()
    exit(0)

    # patch_netvlad_robotcar()
    # exit(0)

    db_fn_path = '/scratch2/fx221/localization/RobotCar-Seasons/3D-models/db_poses.txt'
    query_fn_path = '/scratch2/fx221/localization/RobotCar-Seasons/queries_with_intrinsics_rear.txt'

    all_db_fns = read_fns(path=db_fn_path)
    all_query_fns = read_fns(path=query_fn_path)
    all_query_db_fns = []
    for v in all_db_fns:
        all_query_db_fns.append(v)
    for v in all_query_fns:
        all_query_db_fns.append(v)

    with open('/home/mifs/fx221/Research/Code/shloc/datasets/robotcar/robotcar_db_imglist.txt', 'w') as f:
        
        for v in all_db_fns:
            f.write(v + '\n')

    with open('/home/mifs/fx221/Research/Code/shloc/datasets/robotcar/robotcar_query_imglist.txt', 'w') as f:
        for v in all_query_fns:
            f.write(v + '\n')
    with open('/home/mifs/fx221/Research/Code/shloc/datasets/robotcar/robotcar_db_query_imglist.txt', 'w') as f:
        for v in all_query_db_fns:
            f.write(v + '\n')
    exit(0)

    robotcar_sfm_path = '/scratch2/fx221/localization/RobotCar-Seasons/3D-models/sfm-sift'
    # _, db_images, points3D = read_model(str(robotcar_sfm_path), '.bin')
    # extract_db_poses(all_imapyges=db_images, save_fn='/scratch2/fx221/localization/RobotCar-Seasons/3D-models/db_poses.txt')
    # extract_db_poses(all_images=db_images, save_fn='/scratch2/fx221/localization/RobotCar-Seasons/3D-models/db_poses.txt')

    # pose_from_retrieval()
