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
from tools.common import sort_dict_by_value, resize_img
from localization.tools import read_retrieval_results

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
            # print(osp.join(seg_dir, fn.replace("jpg", "png")))
            if not osp.isfile(osp.join(seg_dir, fn.replace("jpg", "png"))):
                continue
            seg_img = cv2.imread(osp.join(seg_dir, fn.replace("jpg", "png")))
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
                        c_feat = np.load(osp.join(db_feat_dir, c.split('.')[0] + '.npy'), allow_pickle=True).item()[
                            feat_type]
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

    def loc_by_rec_v3_avg_single(self, q_uids, query_feat, db_feats, db_feat_dir, k=5, q_uids_confs=None,
                                 feat_type=None,
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
                        c_feat = np.load(osp.join(db_feat_dir, c.split('.')[0] + '.npy'), allow_pickle=True).item()[
                            feat_type]
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

    def loc_by_rec_v3_avg_cluster(self, query_seg, query_feat, db_feats, db_feat_dir, k=5, q_uids_confs=None,
                                  feat_type=None,
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
                        c_feat = np.load(osp.join(db_feat_dir, c.split('.')[0] + '.npy'), allow_pickle=True).item()[
                            feat_type]
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
