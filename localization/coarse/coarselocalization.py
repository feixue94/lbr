# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> coarselocalization
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   26/07/2021 10:00
=================================================='''
import torch
import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from tools.common import sort_dict_by_value, resize_img
from localization.tools import read_retrieval_results
from tools.seg_tools import label_to_bgr, read_seg_map_without_group


def evaluate(pred_fn, gt_fn):
    results = {}
    pred_retrievals = read_retrieval_results(path=pred_fn)
    gt_retrievals = read_retrieval_results(path=gt_fn)

    for fn in sorted(gt_retrievals.keys()):
        gt_cans = gt_retrievals[fn]
        pred_cans = pred_retrievals[fn]
        n_matches = 0
        for g_c in gt_cans:
            if g_c in pred_cans:
                n_matches += 1
        results[fn] = n_matches
        if n_matches == 0:
            print(fn)
    return results


class CoarseLocalization:
    def __init__(self):
        self.db_list = None
        self.db_feats = None

    def load_db_feat(self, feat_dir, list_fn=None):
        imglist = []
        if list_fn is not None:
            with open(list_fn, "r") as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    imglist.append(l)

        else:
            imglist = os.listdir(feat_dir)

        imglist = sorted(imglist)

        feats = []
        for fn in imglist:
            feati = np.load(osp.join(feat_dir, fn))
            feats.append(feati)
        feats = np.array(feats, np.float)

        self.db_list = imglist
        self.db_feats = feats

        print("Load {:d} feats from {:s}".format(len(imglist), feat_dir))

    def loc_by_feat(self, query_feats, k=20):
        with torch.no_grad():
            query_feat_torch = torch.from_numpy(query_feats).cuda()
            db_feat_torch = torch.from_numpy(self.db_feats).cuda()

            dist = query_feat_torch @ db_feat_torch.t()
            topk = torch.topk(dist, dim=1, k=k, largest=True)[1]

        outputs = []
        for i in range(query_feats.shape[0]):
            results = []
            for j in range(topk.shape[i]):
                results.append(self.db_list[topk[i, j]])
            outputs.append(results)

        return outputs

    def load_db_rec(self, seg_dir, list_fn=None, valid_gid_fn=None, save_tmp_fn=None):
        if save_tmp_fn is not None and osp.exists(save_tmp_fn + ".npy"):
            print("Load data from {:s}".format(save_tmp_fn))
            data = np.load(save_tmp_fn + ".npy", allow_pickle=True).item()
            self.gid_fns = data["gid_fns"]
            self.fn_gids = data["fn_gids"]
            return
        imglist = []
        if list_fn is not None:
            with open(list_fn, "r") as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    imglist.append(l)

        else:
            imglist = os.listdir(seg_dir)
        imglist = sorted(imglist)

        gid_fns = {}
        fn_gids = {}
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        for fn in tqdm(imglist, total=len(imglist)):
            seg_img = cv2.imread(osp.join(seg_dir, fn.replace("jpg", "png")))
            # if not osp.exists(seg_img):
            #     continue
            # seg_img = seg_img[:, seg_img.shape[1] // 3:seg_img.shape[1] // 3 * 2, ]
            cv2.imshow("img", seg_img)
            cv2.waitKey(5)
            seg_img = cv2.resize(seg_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            gids = np.int32(seg_img[:, :, 2]) * 256 * 256 + np.int32(seg_img[:, :, 1]) * 256 + np.int32(
                seg_img[:, :, 0])
            uids = np.unique(gids).tolist()

            fni_gid = {}

            for id in uids:
                if id == 0:
                    continue
                if id in gid_fns.keys():
                    gid_fns[id].append(fn)
                else:
                    gid_fns[id] = [fn]

                cnt = np.sum(gids == id)
                fni_gid[id] = cnt

            fn_gids[fn] = fni_gid

            # if len(fn_gids.keys()) >= 100:
            #     break

        self.fn_gids = fn_gids
        self.gid_fns = gid_fns

        print("Load {:d} imgs with {:d} gids".format(len(self.fn_gids), len(self.gid_fns)))

        if save_tmp_fn is not None:
            print("Save data to {:s}".format(save_tmp_fn))
            data = {"fn_gids": fn_gids, "gid_fns": gid_fns}
            np.save(save_tmp_fn, data)

        cv2.destroyAllWindows()

    def loc_by_rec(self, query_seg, k=20):
        seg_img = cv2.resize(query_seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        gids = np.int32(seg_img[:, :, 2]) * 256 * 256 + np.int32(seg_img[:, :, 1]) * 256 + np.int32(
            seg_img[:, :, 0])
        uids = np.unique(gids).tolist()

        can_scores = {}
        # step 1: find candidates with predicted labels
        for id in uids:
            if id == 0:
                continue
            q_cnt = np.sum(gids == id)

            if id in self.gid_fns.keys():
                for c in self.gid_fns[id]:
                    c_cnt = self.fn_gids[c][id]

                    score = min(q_cnt, c_cnt) / max(q_cnt, c_cnt)
                    if c in can_scores.keys():
                        can_scores[c].append(score)
                    else:
                        can_scores[c] = [score]

        # step 2: sort candidates according to the distribution of labels
        for c in can_scores.keys():
            score = np.mean(can_scores[c]) + len(can_scores[c])
            can_scores[c] = score

        sorted_can_scores = sort_dict_by_value(data=can_scores, reverse=True)
        results = []
        for idx, item in enumerate(sorted_can_scores):
            if idx == k:
                break

            results.append(item)

        return results

    def loc_by_rec_v2(self, query_seg, k=30):
        seg_img = cv2.resize(query_seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        gids = np.int32(seg_img[:, :, 2]) * 256 * 256 + np.int32(seg_img[:, :, 1]) * 256 + np.int32(
            seg_img[:, :, 0])
        uids = np.unique(gids).tolist()
        q_uids = [v for v in uids if v > 0]

        can_scores = {}
        can_uids = {}
        can_cnts = {}
        # step 1: find candidates with predicted labels
        for id in q_uids:
            if id == 0:
                continue

            q_cnt = np.sum(gids == id)

            if id in self.gid_fns.keys():
                for c in self.gid_fns[id]:
                    c_cnt = self.fn_gids[c][id]
                    score = min(q_cnt, c_cnt) / max(q_cnt, c_cnt)

                    if c in can_uids.keys():
                        can_uids[c].append(id)
                        can_scores[c].append(score)
                        can_cnts[c].append(c_cnt)
                    else:
                        can_uids[c] = [id]
                        can_scores[c] = [score]
                        can_cnts[c] = [c_cnt]

        # step2: sort candidates according to the distribution of the labels
        two_more = {}
        one = {}
        for c in can_scores.keys():
            if len(can_uids[c]) > 1:
                two_more[c] = np.mean(can_scores[c]) + len(can_scores[c])
            else:
                one[c] = np.mean(can_scores[c]) + len(can_scores[c])

        sorted_two_more_can_scores = sort_dict_by_value(data=two_more, reverse=True)
        results = []
        quid_obs = {}
        for uid in q_uids:
            quid_obs[uid] = 0

        # selecte from >= 2 obs
        obsK = max(5, k // len(q_uids))
        for idx, item in enumerate(sorted_two_more_can_scores):
            c = item[0]
            co_uids = can_uids[c]
            retain = False
            for id in co_uids:
                if quid_obs[id] < obsK:
                    retain = True
                break
            if retain:
                for id in co_uids:
                    quid_obs[id] = quid_obs[id] + 1
                results.append(item)
        # select from 1 obs
        sorted_one_can = sort_dict_by_value(data=one, reverse=True)
        for idx, item in enumerate(sorted_one_can):
            c = item[0]
            # print(c)
            co_uids = can_uids[c]
            retain = False
            for id in co_uids:
                if quid_obs[id] < obsK:
                    retain = True
                break
            if retain:
                for id in co_uids:
                    quid_obs[id] = quid_obs[id] + 1
                results.append(item)

        return results

    def loc_by_rec_v3(self, query_seg, query_feat, db_feats, db_feat_dir, k=30, q_uids=None):
        seg_img = cv2.resize(query_seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        gids = np.int32(seg_img[:, :, 2]) * 256 * 256 + np.int32(seg_img[:, :, 1]) * 256 + np.int32(
            seg_img[:, :, 0])
        uids = np.unique(gids).tolist()

        if q_uids is None:
            q_uids = [v for v in uids if v > 0]

        can_fns = []
        # step 1: find candidates with predicted labels
        fn_couids = {}
        for id in q_uids:
            if id == 0:
                continue

            if id in self.gid_fns.keys():
                for c in self.gid_fns[id]:
                    if not c in can_fns:
                        can_fns.append(c)
                    if not c in fn_couids.keys():
                        fn_couids[c] = 1
                    else:
                        fn_couids[c] = fn_couids[c] + 1

        # step2: sort candidates according to the distribution of the labels
        can_feats = []
        can_nuid = []
        for c in can_fns:
            if c in db_feats.keys():
                c_feat = db_feats[c]
                can_feats.append(c_feat)
            else:
                c_feat = np.load(osp.join(db_feat_dir, c.split('.')[0] + '.npy'), allow_pickle=True).item()['feat_max']
                # c_feat = c_feat - np.mean(c_feat)
                c_norm = np.linalg.norm(c_feat)
                c_feat = c_feat / c_norm
                can_feats.append(c_feat)
                db_feats[c] = c_feat

            can_nuid.append(fn_couids[c])
        can_nuid_torch = torch.from_numpy(np.array(can_nuid, np.float)).cuda()
        query_feat_torch = torch.from_numpy(query_feat).unsqueeze(0).cuda().float()
        can_feats_torch = torch.from_numpy(np.array(can_feats, np.float)).float().t().cuda()
        print(query_feat_torch.shape, can_feats_torch.shape)
        dist = query_feat_torch @ can_feats_torch
        dist = dist  # + can_nuid_torch.unsqueeze(0)
        topk = min(k, len(can_fns))

        pred_dists, pred_idxs = torch.topk(dist, dim=1, k=topk, largest=True)
        print(pred_idxs.shape, pred_idxs[0, 0])
        results = []
        for i in range(topk):
            results.append((can_fns[pred_idxs[0, i]], pred_dists[0, i].cpu().numpy()))

        return results

    def loc_by_rec_v3_avg(self, query_seg, query_feat, db_feats, db_feat_dir, k=5, q_uids_confs=None, feat_type=None,
                          log_info=None,
                          k_rec=50,
                          ):
        if q_uids_confs is not None:
            sorted_q_uid_confs = sort_dict_by_value(data=q_uids_confs, reverse=True)
            q_uids = []
            for item in sorted_q_uid_confs:
                q_uids.append(item[0])
        else:
            seg_img = cv2.resize(query_seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            gids = np.int32(seg_img[:, :, 2]) * 256 * 256 + np.int32(seg_img[:, :, 1]) * 256 + np.int32(
                seg_img[:, :, 0])
            uids = np.unique(gids).tolist()
            q_uids = [v for v in uids if v > 0]

        # step 1: find candidates with predicted labels
        fn_couids = {}

        results = []
        retrieval_fns = []
        query_feat_torch = torch.from_numpy(query_feat).unsqueeze(0).cuda().float()

        for idx, id in enumerate(q_uids):
            b = id % 256
            g = (id // 256) % 256
            r = id // (256 * 256)
            # print('rgb: ', r, g, b, q_uids_confs[id], idx, len(q_uids))
            text = 'rgb: {:d} {:d} {:d}, conf: {:.6f}, order: {:d}/{:d}'.format(r, g, b, q_uids_confs[id], idx,
                                                                                len(q_uids))
            # print(text)

            if log_info is not None:
                log_info += (text + '\n')
            if id == 0:
                continue
            can_feats = []
            can_fns = []

            if id in self.gid_fns.keys():
                for c in self.gid_fns[id]:
                    if not c in fn_couids.keys():
                        fn_couids[c] = 1
                    else:
                        fn_couids[c] = fn_couids[c] + 1

                    if c in retrieval_fns:
                        continue
                    # else:
                    can_fns.append(c)
                    if c in db_feats.keys():
                        c_feat = db_feats[c]
                    else:
                        c_feat = np.load(osp.join(db_feat_dir, c.split('.')[0] + '.npy'), allow_pickle=True).item()[feat_type]
                        # c_feat = c_feat - np.mean(c_feat)
                        c_norm = np.linalg.norm(c_feat)
                        c_feat = c_feat / c_norm
                        db_feats[c] = c_feat
                    can_feats.append(c_feat)
            # print('len_can_feats: ', len(can_feats))
            if len(can_feats) == 0:
                continue
            can_feats_torch = torch.from_numpy(np.array(can_feats, np.float)).float().t().cuda()
            if len(can_feats_torch.shape) == 1:
                can_feats_torch = can_feats_torch.unsqueeze(1)

            # print(query_feat_torch.shape, can_feats_torch.shape)
            dist = query_feat_torch @ can_feats_torch
            topk = min(k, len(can_feats))
            pred_dists, pred_idxs = torch.topk(dist, dim=1, k=topk, largest=True)
            for i in range(topk):
                if can_fns[pred_idxs[0, i]] in retrieval_fns:  # already found
                    continue
                retrieval_fns.append(can_fns[pred_idxs[0, i]])
                results.append((can_fns[pred_idxs[0, i]], pred_dists[0, i].cpu().numpy()))

            del can_feats
            del can_feats_torch
            del dist

            if len(results) >= k_rec:
                break
        return results

    def loc_by_rec_v3_avg_single(self, q_uids, query_feat, db_feats, db_feat_dir, k=5, q_uids_confs=None, feat_type=None,
                          log_info=None):
        # step 1: find candidates with predicted labels
        fn_couids = {}

        results = []
        # retrieval_fns = []
        query_feat_torch = torch.from_numpy(query_feat).unsqueeze(0).cuda().float()

        for idx, id in enumerate(q_uids):
            b = id % 256
            g = (id // 256) % 256
            r = id // (256 * 256)
            # print('rgb: ', r, g, b, q_uids_confs[id], idx, len(q_uids))
            text = 'rgb: {:d} {:d} {:d}, conf: {:.6f}, order: {:d}/{:d}'.format(r, g, b, q_uids_confs[id], idx,
                                                                                len(q_uids))
            print(text)

            if log_info is not None:
                log_info += (text + '\n')
            if id == 0:
                continue
            can_feats = []
            can_fns = []

            if id in self.gid_fns.keys():
                for c in self.gid_fns[id]:
                    if not c in fn_couids.keys():
                        fn_couids[c] = 1
                    else:
                        fn_couids[c] = fn_couids[c] + 1

                    # if c in retrieval_fns:
                    #     continue
                    # else:
                    can_fns.append(c)
                    if c in db_feats.keys():
                        c_feat = db_feats[c]
                    else:
                        c_feat = np.load(osp.join(db_feat_dir, c.split('.')[0] + '.npy'), allow_pickle=True).item()[feat_type]
                        # c_feat = c_feat - np.mean(c_feat)
                        c_norm = np.linalg.norm(c_feat)
                        c_feat = c_feat / c_norm
                        db_feats[c] = c_feat
                    can_feats.append(c_feat)
            # print('len_can_feats: ', len(can_feats))
            can_feats_torch = torch.from_numpy(np.array(can_feats, np.float)).float().t().cuda()
            if len(can_feats_torch.shape) == 1:
                can_feats_torch = can_feats_torch.unsqueeze(1)

            print(query_feat_torch.shape, can_feats_torch.shape)
            dist = query_feat_torch @ can_feats_torch
            topk = min(k, len(can_feats))
            pred_dists, pred_idxs = torch.topk(dist, dim=1, k=topk, largest=True)
            for i in range(topk):
                # if can_fns[pred_idxs[0, i]] in retrieval_fns:  # already found
                #     continue
                # retrieval_fns.append(can_fns[pred_idxs[0, i]])
                results.append((can_fns[pred_idxs[0, i]], pred_dists[0, i].cpu().numpy()))
        return results

    def loc_by_rec_v3_avg_cluster(self, query_seg, query_feat, db_feats, db_feat_dir, k=5, q_uids_confs=None, feat_type=None,
                          log_info=None, k_rec=50):
        seg_img = cv2.resize(query_seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        gids = np.int32(seg_img[:, :, 2]) * 256 * 256 + np.int32(seg_img[:, :, 1]) * 256 + np.int32(
            seg_img[:, :, 0])
        uids = np.unique(gids).tolist()

        if q_uids_confs is not None:
            sorted_q_uid_confs = sort_dict_by_value(data=q_uids_confs, reverse=True)
            q_uids = []
            for item in sorted_q_uid_confs:
                q_uids.append(item[0])
        else:
            q_uids = [v for v in uids if v > 0]

        # step 1: find candidates with predicted labels
        fn_couids = {}

        results = []
        # retrieval_fns = []
        query_feat_torch = torch.from_numpy(query_feat).unsqueeze(0).cuda().float()

        for idx, id in enumerate(q_uids):
            b = id % 256
            g = (id // 256) % 256
            r = id // (256 * 256)
            # print('rgb: ', r, g, b, q_uids_confs[id], idx, len(q_uids))
            text = 'rgb: {:d} {:d} {:d}, conf: {:.6f}, order: {:d}/{:d}'.format(r, g, b, q_uids_confs[id], idx,
                                                                                len(q_uids))
            # print(text)

            if log_info is not None:
                log_info += (text + '\n')
            if id == 0:
                continue
            can_feats = []
            can_fns = []

            if id in self.gid_fns.keys():
                for c in self.gid_fns[id]:
                    if c not in fn_couids.keys():
                        fn_couids[c] = 1
                    else:
                        fn_couids[c] = fn_couids[c] + 1

                    # if c in retrieval_fns:
                    #     continue
                    # else:
                    can_fns.append(c)
                    if c in db_feats.keys():
                        c_feat = db_feats[c]
                    else:
                        c_feat = np.load(osp.join(db_feat_dir, c.split('.')[0] + '.npy'), allow_pickle=True).item()[feat_type]
                        # c_feat = c_feat - np.mean(c_feat)
                        c_norm = np.linalg.norm(c_feat)
                        c_feat = c_feat / c_norm
                        db_feats[c] = c_feat
                    can_feats.append(c_feat)
            # print('len_can_feats: ', len(can_feats))

            if len(can_feats) == 0:
                continue

            can_feats_torch = torch.from_numpy(np.array(can_feats, np.float)).float().t().cuda()
            if len(can_feats_torch.shape) == 1:
                can_feats_torch = can_feats_torch.unsqueeze(1)

            # print(query_feat_torch.shape, can_feats_torch.shape)
            dist = query_feat_torch @ can_feats_torch
            topk = min(k, len(can_feats))
            pred_dists, pred_idxs = torch.topk(dist, dim=1, k=topk, largest=True)

            sel_cans = []
            for i in range(topk):
                # if can_fns[pred_idxs[0, i]] in retrieval_fns:  # already found
                #     continue
                # retrieval_fns.append(can_fns[pred_idxs[0, i]])
                sel_cans.append((can_fns[pred_idxs[0, i]], pred_dists[0, i].cpu().numpy()))

            results.append(sel_cans)

            del can_feats
            del can_feats_torch
            del dist
            del pred_idxs
            del pred_dists

            if len(results) >= k_rec:
                break

        return results

    def loc_by_rec_v4_avg(self, query_seg, query_feat, db_feats, db_feat_dir, k=5, q_uids_confs=None, feat_type=None):
        if q_uids_confs is not None:
            sorted_q_uid_confs = sort_dict_by_value(data=q_uids_confs, reverse=True)
            q_uids = []
            for item in sorted_q_uid_confs:
                q_uids.append(item[0])
        else:
            seg_img = cv2.resize(query_seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            gids = np.int32(seg_img[:, :, 2]) * 256 * 256 + np.int32(seg_img[:, :, 1]) * 256 + np.int32(
                seg_img[:, :, 0])
            uids = np.unique(gids).tolist()
            q_uids = [v for v in uids if v > 0]

        results = []
        query_feat_torch = torch.from_numpy(query_feat).unsqueeze(0).cuda().float()

        can_feats = []
        can_fns = []
        for id in q_uids:
            b = id % 256
            g = (id // 256) % 256
            r = id // (256 * 256)
            print('rgb: ', r, g, b, q_uids_confs[id])
            if id == 0:
                continue
            if id in self.gid_fns.keys():
                for c in self.gid_fns[id]:
                    if c in can_fns:
                        continue
                    can_fns.append(c)
                    if c in db_feats.keys():
                        c_feat = db_feats[c]
                    else:
                        c_feat = np.load(osp.join(db_feat_dir, c.split('.')[0] + '.npy'), allow_pickle=True).item()[
                            feat_type]
                        # c_feat = c_feat - np.mean(c_feat)
                        c_norm = np.linalg.norm(c_feat)
                        c_feat = c_feat / c_norm
                        db_feats[c] = c_feat
                    can_feats.append(c_feat)

        print('can_feats: ', len(can_feats))
        can_feats_torch = torch.from_numpy(np.array(can_feats, np.float)).float().t().cuda()
        if len(can_feats_torch.shape) == 1:
            can_feats_torch = can_feats_torch.unsqueeze(1)

        print(query_feat_torch.shape, can_feats_torch.shape)
        dist = query_feat_torch @ can_feats_torch
        topk = min(k, len(can_feats))
        pred_dists, pred_idxs = torch.topk(dist, dim=1, k=topk, largest=True)
        for i in range(topk):
            results.append((can_fns[pred_idxs[0, i]], pred_dists[0, i].cpu().numpy()))
        return results


def prediction_to_labels(pred_conf, pred_labels, cnt_th=2000, cnt_labels=5, map_gid_rgb=None):
    topk = pred_conf.shape[0]
    final_label = np.zeros(shape=(256, 256), dtype=np.int)
    final_conf = np.zeros(shape=(256, 256), dtype=np.float)
    nlabels = 0
    q_uids = []

    puid_conf = {}
    for i in range(topk):
        label = pred_labels[i]
        conf = pred_conf[i]
        conf = cv2.resize(conf, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        mask_conf = (conf < 1e-4)
        label[mask_conf] = 0
        # seg = label_to_bgr(label=label, maps=map_gid_rgb)
        uids = np.unique(label).tolist()
        valid_uids = [v for v in uids if v > 0]
        # print('i: pred_labels: ', i, valid_uids)
        gid_conf = {}

        for v in valid_uids:
            if v in q_uids:
                continue
            # cnt = np.sum(label == v)
            cf = np.mean(conf[label == v])
            gid_conf[v] = cf

        sorted_gid_conf = sort_dict_by_value(data=gid_conf, reverse=True)

        new_label = np.zeros_like(label)
        new_conf = np.zeros_like(conf)
        for v, c in sorted_gid_conf:

            if v not in puid_conf.keys():
                puid_conf[v] = c
            if v not in q_uids:
                q_uids.append(v)

            mask = (label == v)
            new_label[mask] = v
            new_conf[mask] = conf[mask]
            nlabels += 1
            if nlabels >= cnt_labels:
                break

        # new_seg = label_to_bgr(label=new_label, maps=map_gid_rgb)
        # cv2.imshow('seg{:d}'.format(i + 1), new_seg)
        # cv2.waitKey(0)

        empty_mask = (final_label == 0)
        final_label[empty_mask] = new_label[empty_mask]
        final_conf[empty_mask] = new_conf[empty_mask]

        cnt = np.sum(final_label > 0)
        # print('cnt: ', cnt)
        fuids = np.unique(final_label).tolist()
        fuids = [v for v in fuids if v > 0]
        nuids = len(fuids)

        # if i == 0:
        #     if cnt > cnt_th:
        #         print("topk: ", i, cnt, nuids)
        #         return final_conf, final_label, q_uids, puid_conf
        # else:
        # if cnt > cnt_th and nuids >= cnt_labels:
        if nuids >= cnt_labels:
            print("topk: ", i, cnt, nuids)
            return final_conf, final_label, q_uids, puid_conf
    return final_conf, final_label, q_uids, puid_conf


def prediction_to_labels_v2(pred_conf, pred_labels, cnt_th=2000, cnt_labels=5, map_gid_rgb=None):
    topk = pred_conf.shape[0]
    puid_conf = {}
    puid_mask = {}
    for i in range(topk):
        label = pred_labels[i]
        conf = pred_conf[i]
        conf = cv2.resize(conf, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        mask_conf = (conf < 1e-4)
        label[mask_conf] = 0
        # seg = label_to_bgr(label=label, maps=map_gid_rgb)
        uids = np.unique(label).tolist()
        valid_uids = [v for v in uids if v > 0]

        for v in valid_uids:
            mask = (label == v)
            cf = np.mean(conf[mask])
            if v in puid_conf.keys():
                if cf > puid_conf[v]:
                    puid_conf[v] = cf
                    puid_mask[v] = mask
            else:
                puid_conf[v] = cf
                puid_mask[v] = mask

    print('Find {:d} global instances'.format(len(puid_conf.keys())))

    final_label = np.zeros(shape=(256, 256), dtype=np.int)
    final_conf = np.zeros(shape=(256, 256), dtype=np.float)
    final_puid_conf = {}
    sorted_puid_conf = sort_dict_by_value(data=puid_conf, reverse=True)
    final_uids = []
    for v in sorted_puid_conf:
        uid = v[0]
        cf = v[1]
        mask_v = puid_mask[uid]
        final_puid_conf[uid] = cf

        mask = (final_label == 0) * mask_v
        final_label[mask] = uid
        final_conf[mask] = cf

        final_uids.append(uid)

        # if len(final_uids) >= cnt_labels:
        #     break

    return final_conf, final_label, final_uids, final_puid_conf


def prediction_to_labels_v3(pred_conf, pred_labels, cnt_th=2000, cnt_labels=5, map_gid_rgb=None):
    topk = pred_conf.shape[0]
    puid_conf = {}
    puid_mask = {}
    for i in range(topk):
        label = pred_labels[i]
        conf = pred_conf[i]
        conf = cv2.resize(conf, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        mask_conf = (conf < 1e-4)
        label[mask_conf] = 0
        # seg = label_to_bgr(label=label, maps=map_gid_rgb)
        uids = np.unique(label).tolist()
        valid_uids = [v for v in uids if v > 0]

        for v in valid_uids:
            mask = (label == v)
            cf = np.mean(conf[mask]) + topk - i
            if v in puid_conf.keys():
                if cf > puid_conf[v]:
                    puid_conf[v] = cf
                    puid_mask[v] = mask
            else:
                puid_conf[v] = cf
                puid_mask[v] = mask

    print('Find {:d} global instances'.format(len(puid_conf.keys())))

    final_label = np.zeros(shape=(256, 256), dtype=np.int)
    final_conf = np.zeros(shape=(256, 256), dtype=np.float)
    final_puid_conf = {}
    sorted_puid_conf = sort_dict_by_value(data=puid_conf, reverse=True)
    final_uids = []
    for v in sorted_puid_conf:
        uid = v[0]
        cf = v[1]
        mask_v = puid_mask[uid]
        final_puid_conf[uid] = cf

        mask = (final_label == 0) * mask_v
        final_label[mask] = uid
        final_conf[mask] = cf

        final_uids.append(uid)

        # if len(final_uids) >= cnt_labels:
        #     break

    return final_conf, final_label, final_uids, final_puid_conf


def retrieval_aachen():
    save_root = "/data/cornucopia/fx221/exp/shloc/aachen"
    img_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright"

    # feat_type = 'feat_max'
    # feat_type = 'netvlad_512_hard'
    # feat_type = 'netvlad_512_hard_atten'
    feat_type = 'gem_hard'
    # feat_type = 'gem_hard_atten'
    k_seg = 10
    k_can = 10
    version = 'v6'
    show_img = 5
    pversion = 3
    rversion = 3
    save_pred_seg = False
    if version == 'v3':
        map_gid_rgb = read_seg_map_without_group("datasets/aachen/aachen_grgb_gid_v3.txt")
        db_seg_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/global_seg_instance/merge_v3"
        # weight_name = "2021_08_03_11_06_14_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized"
        weight_name = "2021_08_05_23_29_59_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized"
        db_imglist_fn = "datasets/aachen/aachen_train_file_list_full_v3.txt"
        save_tmp_fn = 'aachen_517'

        save_fn = 'aachen_loc_by_seg_517_l{:d}_top{:d}.txt'.format(k_seg, k_can)
    elif version == 'v4':
        map_gid_rgb = read_seg_map_without_group("datasets/aachen/aachen_grgb_gid_v4.txt")
        db_seg_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/global_seg_instance"
        # weight_name = '2021_08_10_23_09_09_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized'
        # weight_name = '2021_08_10_23_08_23_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized_full'
        # weight_name = '2021_08_11_22_59_34_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized'
        weight_name = '2021_08_11_23_01_03_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug12_stylized'
        # weight_name = '2021_08_11_23_01_45_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized_full'
        db_imglist_fn = "datasets/aachen/aachen_train_file_list_full_v4.txt"
        save_tmp_fn = 'aachen_481'
    elif version == 'v5':
        map_gid_rgb = read_seg_map_without_group("datasets/aachen/aachen_grgb_gid_v5.txt")
        db_seg_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/global_seg_instance"
        # weight_name = '2021_08_14_17_57_21_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized'
        # weight_name = '2021_08_14_18_11_25_aachen_pspf_resnext101_32x4d_d4_u8_ce_b16_R256_seg_cls_aug_stylized'
        # weight_name = '2021_08_18_11_49_22_aachen_pspf_resnext101_32x4d_d4_u8_b24_R256_ceohem_adamw_seg_cls_aug_stylized'
        # weight_name = '2021_08_18_17_53_46_aachen_pspf_resnext101_32x4d_d4_u8_b24_R256_ceohem_adam_seg_cls_aug_stylized'
        # weight_name = '2021_08_19_23_50_04_aachen_pspf_resnext101_32x4d_d4_u8_b24_R256_ce_adam_seg_cls_aug_stylized'
        # weight_name = '2021_08_20_12_58_52_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_ceohem_adam_seg_cls_aug_stylized'
        # weight_name = '2021_08_22_12_03_06_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_ceohem_adam_seg_cls_aug_stylized/512'
        weight_name = '2021_08_23_18_10_31_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_E120_ceohem_adam_seg_cls_aug_stylized'
        db_imglist_fn = "datasets/aachen/aachen_train_file_list_full_v5.txt"
        save_tmp_fn = 'aachen_451'

        # save_fn = 'aachen_loc_by_seg_451_{:s}_l{:d}_top{:d}_v2_failed_cases_b16.txt'.format(feat_type, k_seg, k_can)
        # save_fn = 'aachen_loc_by_seg_451_{:s}_l{:d}_top{:d}_v3_failed_cases_b16.txt'.format(feat_type, k_seg, k_can)
        # save_fn = 'aachen_loc_by_seg_451_{:s}_l{:d}_top{:d}_p3_r3.txt'.format(feat_type, k_seg, k_can)
        save_fn = 'aachen_loc_by_seg_451_{:s}_l{:d}_top{:d}_p{:d}_r{:d}.txt'.format(feat_type, k_seg, k_can, pversion,
                                                                                    rversion)
        # save_fn = 'aachen_loc_by_seg_451_{:s}_l{:d}_top{:d}_p{:d}_r{:d}_for_order_more_than_1_512.txt'.format(feat_type, k_seg, k_can, pversion, rversion)
    elif version == 'v6':
        map_gid_rgb = read_seg_map_without_group("datasets/aachen/aachen_grgb_gid_v5.txt")
        db_seg_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/global_seg_instance"
        # weight_name = '2021_08_26_22_18_23_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_E120_ceohem_adam_seg_cls_aug_stylized'
        # weight_name = '2021_08_28_00_45_02_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_E120_ceohem_adam_seg_cls_aug_stylized'
        weight_name = '2021_08_29_12_49_48_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_E120_ceohem_adam_seg_cls_aug_stylized'
        db_imglist_fn = "datasets/aachen/aachen_train_file_list_full_v5.txt"
        save_tmp_fn = 'aachen_452'

        # save_fn = 'aachen_loc_by_seg_451_{:s}_l{:d}_top{:d}_v2_failed_cases_b16.txt'.format(feat_type, k_seg, k_can)
        # save_fn = 'aachen_loc_by_seg_451_{:s}_l{:d}_top{:d}_v3_failed_cases_b16.txt'.format(feat_type, k_seg, k_can)
        # save_fn = 'aachen_loc_by_seg_451_{:s}_l{:d}_top{:d}_p3_r3.txt'.format(feat_type, k_seg, k_can)
        save_fn = 'aachen_loc_by_seg_452_{:s}_l{:d}_top{:d}_p{:d}_r{:d}_gem_hard_failed_cases.txt'.format(feat_type,
                                                                                                          k_seg, k_can,
                                                                                                          pversion,
                                                                                                          rversion)

    q_pred_dir = osp.join(save_root, weight_name, "confidence")
    save_seg_dir = osp.join(save_root, weight_name, "masks")
    # q_imglist_fn = "datasets/aachen/aachen_query_imglist.txt"
    # q_imglist_fn = "datasets/aachen/fail_case_aachen_nv.txt"
    # q_imglist_fn = "datasets/aachen/fail_imglist.txt"
    # q_imglist_fn = "failed_cases_451_rec_b16.txt"
    # q_imglist_fn = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_20_12_58_52_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_ceohem_adam_seg_cls_aug_stylized/loc_by_seg/aachen_loc_by_seg_451_feat_max_l10_top5_v3_failed_cases_b16_th8.txt"
    # q_imglist_fn = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_20_12_58_52_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_ceohem_adam_seg_cls_aug_stylized/loc_by_seg/failed_cases_netvlad_l10_top5_v3_opt_th10.txt"
    # q_imglist_fn = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_22_12_03_06_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_ceohem_adam_seg_cls_aug_stylized/loc_by_seg/r2d2-rmax1600-10k-aachen_loc_by_seg_451_feat_max_l10_top5_p3_r3_sinlge_v3_th10_order_more_than_1.txt"
    # q_imglist_fn = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_23_18_10_31_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_E120_ceohem_adam_seg_cls_aug_stylized/loc_by_seg/r2d2-rmax1600-10k-aachen_loc_by_seg_451_netvlad_l10_top5_p3_r3_th10.0.txt.failed"
    q_imglist_fn = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_29_12_49_48_aachen_pspf_resnext101_32x4d_d4_u8_b16_R256_E120_ceohem_adam_seg_cls_aug_stylized/loc_by_seg/r2d2-rmax1600-10k-aachen_loc_by_seg_452_gem_hard_l10_top5_p3_r3_th10.0.txt.failed"
    save_dir = osp.join(save_root, weight_name, "loc_by_seg")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    debugging = False
    q_imglist = []
    with open(q_imglist_fn, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            q_imglist.append(l)

    Localizer = CoarseLocalization()
    Localizer.load_db_rec(seg_dir=db_seg_dir, list_fn=db_imglist_fn,
                          save_tmp_fn=save_tmp_fn)

    results = {}
    next_q = False
    if feat_type.find('netvlad') >= 0 or feat_type.find('gem') >= 0:
        print('Load global feats from {:s}'.format(osp.join(osp.join(save_root, weight_name, feat_type + '.npy'))))
        use_global_feat = True
        global_feats = np.load(osp.join(save_root, weight_name, feat_type + '.npy'), allow_pickle=True).item()
        db_feats = global_feats
    else:
        use_global_feat = False
        db_feats = {}
    for idx, q_fn in enumerate(tqdm(q_imglist, total=len(q_imglist))):
        # if q_fn in q_db_pairs_gt.keys():
        #     continue
        # if idx <= 17:
        #     continue

        # if q_fn.find('2011-12-17_15-00-40_564') < -1:
        #     continue
        print('q_fn: ', q_fn)
        next_q = False
        # q_seg = cv2.imread(osp.join(q_seg_dir, q_fn.replace("jpg", "png")))
        pred = np.load(osp.join(q_pred_dir, q_fn.split(".")[0] + ".npy"), allow_pickle=True).item()
        # print(pred.keys())
        pred_confs = pred["confidence"]

        pred_labels = pred["ids"]

        if use_global_feat:
            q_feat = global_feats[q_fn]
        else:
            q_feat = pred[feat_type]

        # q_feat = q_feat - np.mean(q_feat)
        q_feat = q_feat / np.linalg.norm(q_feat)
        if pversion == 2:
            q_f, q_label, q_uids, q_uid_confs = prediction_to_labels_v2(pred_conf=pred_confs, pred_labels=pred_labels,
                                                                        cnt_th=5000,
                                                                        cnt_labels=k_seg,
                                                                        map_gid_rgb=map_gid_rgb)
        elif pversion == 3:
            q_f, q_label, q_uids, q_uid_confs = prediction_to_labels_v3(pred_conf=pred_confs, pred_labels=pred_labels,
                                                                        cnt_th=5000,
                                                                        cnt_labels=k_seg,
                                                                        map_gid_rgb=map_gid_rgb)
        # q_seg = q_seg[:, q_seg.shape[1] // 3:q_seg.shape[1] // 3 * 2, ]
        q_seg = label_to_bgr(label=q_label, maps=map_gid_rgb)

        q_rgb_confs = {}
        for v in q_uid_confs.keys():
            q_rgb_confs[map_gid_rgb[v]] = q_uid_confs[v]

        if show_img > 0:
            q_img = cv2.imread(osp.join(img_dir, q_fn))
            q_img = resize_img(img=q_img, nh=256)
            cv2.namedWindow("q_img", cv2.WINDOW_NORMAL)
            cv2.namedWindow("q_seg", cv2.WINDOW_NORMAL)
            cv2.imshow("q_img", q_img)
            cv2.imshow("q_seg", q_seg)
            # cv2.waitKey(0)
            # continue

        if save_pred_seg:
            cv2.imwrite(osp.join(save_seg_dir, q_fn.replace("jpg", "png")), q_seg)

        # cands = Localizer.loc_by_rec_v2(query_seg=q_seg, k=50)
        # cands = Localizer.loc_by_rec_v3(query_seg=q_seg, query_feat=q_feat, db_feats=db_feats, db_feat_dir=q_pred_dir,
        #                                 k=k_can)
        if rversion == 3:
            cands = Localizer.loc_by_rec_v3_avg(query_seg=q_seg, query_feat=q_feat, db_feats=db_feats,
                                                db_feat_dir=q_pred_dir,
                                                k=k_can,
                                                q_uids_confs=q_rgb_confs,
                                                feat_type=feat_type)
        elif rversion == 4:
            cands = Localizer.loc_by_rec_v4_avg(query_seg=q_seg, query_feat=q_feat, db_feats=db_feats,
                                                db_feat_dir=q_pred_dir,
                                                k=100,
                                                q_uids_confs=q_rgb_confs,
                                                feat_type=feat_type)
        # cands = q_db_pairs_nv[q_fn]
        print("Find {:d} candidates".format(len(cands)))

        results[q_fn] = []
        for idx, c_item in enumerate(cands):
            print(c_item)
            if next_q:
                break

            # filter low confidence
            if c_item[1] < 0.5:
                continue

            results[q_fn].append(c_item[0])

            if show_img > 0:
                rec_cand_img = cv2.imread(osp.join(img_dir, c_item[0]))
                rec_cand_seg = cv2.imread(osp.join(db_seg_dir, c_item[0].replace("jpg", "png")))
                if rec_cand_seg is None:
                    rec_cand_seg = np.zeros_like(rec_cand_img)
                rec_cand_img = resize_img(img=rec_cand_img, nh=256)
                rec_cand_seg = resize_img(img=rec_cand_seg, nh=256)
                font = cv2.FONT_HERSHEY_SIMPLEX
                rec_cand_img = cv2.putText(rec_cand_img, '{:d}/{:.5f}'.format(idx + 1, c_item[1]), (50, 30), font, 1.,
                                           (0, 0, 255), 2)
                cv2.namedWindow("rec_img", cv2.WINDOW_NORMAL)
                cv2.namedWindow("rec_seg", cv2.WINDOW_NORMAL)
                cv2.imshow("rec_img", rec_cand_img)
                cv2.imshow("rec_seg", rec_cand_seg)
                cv2.waitKey(show_img)

            if len(results[q_fn]) >= 100:
                break

            if debugging:
                while True:
                    cv2.namedWindow("rec_img", cv2.WINDOW_NORMAL)
                    cv2.namedWindow("rec_seg", cv2.WINDOW_NORMAL)
                    cv2.imshow("rec_img", rec_cand_img)
                    cv2.imshow("rec_seg", rec_cand_seg)
                    key = cv2.waitKey()
                    if key == ord("n"):
                        next_q = True
                        break
                    elif key == ord("c"):
                        break
                    elif key == ord("q"):
                        exit(0)
                    elif key == ord('s'):
                        results[q_fn].append(c_item[0])
                        break

    if debugging:
        exit(0)
    with open(osp.join(save_dir, save_fn), "w") as f:
        for fn in results.keys():
            cands = results[fn]
            for c in cands:
                text = fn + " " + c
                f.write(text + "\n")


if __name__ == '__main__':
    pairs_nv = "/home/mifs/fx221/Research/Code/Hierarchical-Localization/pairs/aachen_v1.1/pairs-query-netvlad50.txt"
    pairs_gt = "/home/mifs/fx221/Research/Code/shloc/datasets/aachen/aachen_query_pairs_gt.txt"
    pairs_gt_extend = "/home/mifs/fx221/Research/Code/shloc/datasets/aachen/aachen_query_pairs_gt_extend.txt"
    pairs_db = "/home/mifs/fx221/Research/Code/shloc/datasets/aachen/pairs-db-covis20.txt"
    pairs_rec = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_05_23_29_59_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized/loc_by_seg/aachen_loc_by_seg_517_l10_top30.txt"

    # eval_result = evaluate(pred_fn=pairs_rec, gt_fn=pairs_gt)
    # with open('fail_list_v3.txt', 'w') as f:
    #     for fn in eval_result.keys():
    #         m = eval_result[fn]
    #         if m == 0:
    #             f.write(fn + '\n')
    # exit(0)

    db_pairs = read_retrieval_results(path=pairs_db)
    # q_db_pairs_nv = read_retrieval_results(path=pairs_nv)
    q_db_pairs_gt = read_retrieval_results(path=pairs_gt)

    retrieval_aachen()
    exit(0)

    img_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright"
    save_dir = "/data/cornucopia/fx221/exp/shloc/aachen/2021_08_05_23_29_59_aachen_pspf_resnext101_32x4d_d4_u8_ce_b8_seg_cls_aug_stylized"
    pairs_rec = osp.join(save_dir, "loc_by_seg", "loc_by_sec_top30.txt")
    pairs_nv = "/home/mifs/fx221/Research/Code/Hierarchical-Localization/pairs/aachen_v1.1/pairs-query-netvlad50.txt"

    # cv2.namedWindow('q_img', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('c_img', cv2.WINDOW_NORMAL)

    # wrong_list = []
    # for q in tqdm(q_db_pairs_gt.keys(), total=len(q_db_pairs_gt.keys())):
    #     print('q: ', q)
    #     next_q = False
    #     q_img = cv2.imread(osp.join(img_dir, q))
    #     q_img = resize_img(q_img, nh=480)
    #     cv2.imshow('q_img', q_img)
    #     cans = q_db_pairs_gt[q]
    #     for c in cans:
    #         print('c: ', c)
    #         if next_q:
    #             break
    #         c_img = cv2.imread(osp.join(img_dir, c))
    #         c_img = resize_img(c_img, nh=480)
    #         while True:
    #             cv2.imshow('c_img', c_img)
    #             k = cv2.waitKey()
    #
    #             if k == ord('c'):
    #                 break
    #             elif k == ord('n'):
    #                 next_q = True
    #                 break
    #             elif k == ord('s'):
    #                 wrong_list.append((q, c))
    #                 break
    # exit(0)

    q_db_pairs_nv = read_retrieval_results(path=pairs_nv)
    q_db_pairs_rec = read_retrieval_results(path=pairs_rec)

    print("Load {:d} queries for rec".format(len(q_db_pairs_rec.keys())))
    print("Load {:d} queries for nv".format(len(q_db_pairs_nv.keys())))

    cv2.namedWindow("q", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
    cv2.namedWindow("rec", cv2.WINDOW_NORMAL)
    cv2.namedWindow("nv", cv2.WINDOW_NORMAL)

    failed_cases = []
    q_imglist_fn = "datasets/aachen/fail_case_aachen_nv.txt"
    with open(q_imglist_fn, "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            failed_cases.append(l)

    for q_fn in q_db_pairs_rec.keys():
        # print(q_fn)
        if q_fn not in failed_cases:
            continue
        print("q_fn: ", q_fn)
        # if q_fn.find("day") > 0:
        #     continue
        can_rec = q_db_pairs_rec[q_fn]
        can_nv = q_db_pairs_nv[q_fn]

        q_img = cv2.imread(osp.join(img_dir, q_fn))
        q_pred = resize_img(q_img, nh=512)
        # q_pred = cv2.imread(osp.join(save_dir, "vis", q_fn.replace("jpg", "png")))
        # q_pred = resize_img(q_pred, nh=256)

        for ci in range(10):
            c_img_rec = cv2.imread(osp.join(img_dir, can_rec[ci]))
            c_img_nv = cv2.imread(osp.join(img_dir, can_nv[ci]))
            # q_img = resize_img(q_img, nh=256)
            c_img_rec = resize_img(c_img_rec, nh=256)
            c_img_nv = resize_img(c_img_nv, nh=256)
            # cv2.putText(c_img_nv, "c:{:d}".format(ci + 1), )
            font = cv2.FONT_HERSHEY_SIMPLEX
            # c_img_rec = cv2.putText(c_img_rec, 'c: {:d}'.format(ci + 1), (50, 50), font, 1.2,
            #                         (0, 0, 255), 2)
            # c_img_nv = cv2.putText(c_img_nv, 'c: {:d}'.format(ci + 1), (50, 50), font, 1.2,
            #                        (0, 0, 255), 2)

            cv2.imshow("q", q_pred)
            cv2.imshow("rec", c_img_rec)
            cv2.imshow("nv", c_img_nv)
            cv2.waitKey(0)
