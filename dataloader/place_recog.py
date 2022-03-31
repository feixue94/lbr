# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> global
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   23/08/2021 20:47
=================================================='''
import torch.utils.data as data
import os.path as osp
import cv2
import random
import numpy as np
import torch
from tqdm import tqdm


class PlaceRecog(data.Dataset):
    def __init__(self,
                 image_path,
                 cats,
                 positive_pairs,
                 train=True,
                 transform=None,
                 img_list=None,
                 aug=None,
                 label_path=None,
                 grgb_gid_file=None,
                 n_classes=452,
                 use_cls=False,
                 R=256,
                 n_pos=6,
                 n_neg=6,
                 preload=False,
                 keep_ratio=False,
                 ):
        super(PlaceRecog, self).__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.cls = use_cls
        self.classes = n_classes
        self.transform = transform
        self.train = train
        self.aug = aug
        self.cats = cats
        self.R = R
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.imgs = None
        self.segs = None
        self.preload = preload
        self.keep_ratio = keep_ratio

        # self.valid_fns = []
        self.all_fns = []
        with open(img_list, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                # self.valid_fns.append(l)
                self.all_fns.append(l)

        # print(self.all_fns[0])
        print("Load positive pairs....")
        self.positive_pairs = {}
        with open(positive_pairs, "r") as f:
            lines = f.readlines()
            for l in tqdm(lines, total=len(lines)):
                l = l.strip().split()

                # print(l)
                # exit(0)

                if l[0] not in self.all_fns:
                    continue
                if l[1] not in self.all_fns:
                    continue
                if l[0] in self.positive_pairs.keys():
                    if not l[1] in self.positive_pairs[l[0]]:
                        self.positive_pairs[l[0]].append(l[1])
                else:
                    self.positive_pairs[l[0]] = [l[1]]

                if l[1] in self.positive_pairs.keys():
                    if not l[0] in self.positive_pairs[l[1]]:
                        self.positive_pairs[l[1]].append(l[0])
                else:
                    self.positive_pairs[l[1]] = [l[0]]

        # self.valid_fns = [v for v in self.positive_pairs.keys() if len(self.positive_pairs[v]) >= n_pos]
        print('Positive samples: ', len(self.positive_pairs.keys()))
        if self.cls:
            self.grgb_gid = {}
            with open(grgb_gid_file, "r") as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip().split(" ")
                    self.grgb_gid[int(l[0])] = int(l[1])

        print("Load negative pairs....")

        valid_fns = []
        self.negative_pairs = {}
        for fn in tqdm(self.positive_pairs.keys(), total=len(self.positive_pairs.keys())):
            pos_fns = self.positive_pairs[fn]
            neg_fns = []
            for c in self.all_fns:
                if c in [fn] or c in pos_fns:
                    continue
                neg_fns.append(c)

            self.negative_pairs[fn] = neg_fns

            if len(pos_fns) >= n_pos and len(neg_fns) >= n_neg:
                valid_fns.append(fn)

        self.valid_fns = valid_fns
        # print("Load valid fns....")
        # self.valid_fns = [v for v in self.positive_pairs.keys() if
        #                   v in self.negative_pairs.keys() and len(self.positive_pairs[v]) >= n_pos and len(
        #                       self.negative_pairs[v]) >= n_neg]

        # print("Load all fns....")
        # self.all_fns = []
        # for v in tqdm(self.positive_pairs.keys(), total=len(self.positive_pairs)):
        #     if v not in self.all_fns:
        #         self.all_fns.append(v)
        #
        #     cans = self.positive_pairs[v]
        #     for c in cans:
        #         if c not in self.all_fns:
        #             self.all_fns.append(c)
        #
        # for v in tqdm(self.negative_pairs.keys(), total=len(self.negative_pairs)):
        #     if v not in self.all_fns:
        #         self.all_fns.append(v)
        #
        #     cans = self.negative_pairs[v]
        #     for c in cans:
        #         if c not in self.all_fns:
        #             self.all_fns.append(c)

        self.hard_negative_pairs = None
        self.hard_positive_pairs = None
        self.cls_labels = {}

        # mn_pos = 10000
        # mx_pos = 0
        # mn_neg = 10000
        # mx_neg = 0
        # for v in tqdm(self.valid_fns):
        #     np = len(self.positive_pairs[v])
        #     if np < mn_pos:
        #         mn_pos = np
        #     if np > mx_pos:
        #         mx_pos = np
        #
        #     nn = len(self.negative_pairs[v])
        #     if nn < mn_neg:
        #         mn_neg = nn
        #     if nn > mx_neg:
        #         mx_neg = nn
        #
        # print('mn_pos: {:d} mx_pos: {:d} mn_neg: {:d} mx_neg: {:d}'.format(mn_pos, mx_pos, mn_neg, mx_neg))
        # exit(0)

        if self.preload:
            self.load_images()
    def load_images(self):
        print('Load images with random illumination changes...')
        self.imgs = {}
        for i in range(len(self.cats)):
            self.imgs[i] = {}

        if self.cls:
            if self.segs is not None:
                del self.segs
            self.segs = {}

        for fn in tqdm(self.all_fns):
            for cat_id in range(len(self.cats)):
                img = cv2.imread(osp.join(self.image_path, self.cats[cat_id], fn))
                if self.keep_ratio:
                    size = img.shape[:2][::-1]
                    w, h = size
                    scale = self.R / max(h, w)
                    h_new, w_new = int(round(h * scale)), int(round(w * scale))
                    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

                    img_pad = np.zeros(shape=(self.R, self.R, 3), dtype=np.uint8)
                    img_pad[0:h_new, 0:w_new, :] = img
                    img = img_pad
                else:
                    img = cv2.resize(img, dsize=(self.R, self.R))
                # img = cv2.resize(img, dsize=(self.R, self.R))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.imgs[cat_id][fn] = img

            if self.cls:
                seg_img = cv2.imread(osp.join(self.label_path, fn.replace('jpg', 'png')))
                seg_img = cv2.resize(seg_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
                self.segs[fn] = seg_img

    def seg_to_gid(self, seg_img, grgb_gid_map):
        id_img = np.int32(seg_img[:, :, 2]) * 256 * 256 + np.int32(seg_img[:, :, 1]) * 256 + np.int32(
            seg_img[:, :, 0])
        luids = np.unique(id_img).tolist()
        # print("id_img: ", id_img.shape)
        out_img = np.zeros_like(seg_img)
        gid_img = np.zeros_like(id_img)
        for id in luids:
            if id in grgb_gid_map.keys():
                gid = grgb_gid_map[id]
                mask = (id_img == id)
                gid_img[mask] = gid

                out_img[mask] = seg_img[mask]

        return out_img, gid_img

    def update_negatives(self, model, encoder, dim=512, topk=20, use_atten=False, pre_calc_db_feats=None):
        # Do hard mining
        print('Start hard mining...')
        model.eval()
        all_feats = {}
        # if pre_calc_db_feats is not None:
        #     for fn in pre_calc_db_feats:
        #         all_feats[fn] = pre_calc_db_feats[fn]

        with torch.no_grad():
            print('Extract features...')
            for fn in tqdm(self.all_fns, total=len(self.all_fns)):
                if len(self.cats) > 1:
                    cat_id = random.randint(1, len(self.cats))
                else:
                    cat_id = 1
                if self.imgs is None:
                    img = cv2.imread(osp.join(self.image_path, self.cats[cat_id - 1], fn))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, dsize=(self.R, self.R))
                else:
                    img = self.imgs[cat_id - 1][fn]

                if self.transform is not None:
                    img = self.transform(img).unsqueeze(0).float().cuda()
                with torch.no_grad():
                    if not use_atten:
                        feats = encoder(img)
                        atten = None
                    else:
                        enc_outputs = encoder(img)
                        feats = enc_outputs['feats']
                        masks = enc_outputs['masks'][0]
                        masks = torch.softmax(masks, dim=1)
                        # keep the value of top 20?
                        masks = masks[:, 1:, :, :]
                        values, _ = torch.topk(masks, dim=1, largest=True, k=20)
                        atten = torch.sum(values, dim=1, keepdim=True)
                    if dim == 512:
                        feat = feats[-2]
                    else:
                        feat = feats[-1]

                    feat = model(feat, atten)['feat']
                    all_feats[fn] = feat.reshape(1, -1)

            print('Find hardest negatives...')
            self.hard_negative_pairs = {}
            for q_fn in tqdm(self.valid_fns, total=len(self.valid_fns)):
                q_feat = all_feats[q_fn]
                all_neg_fns = self.negative_pairs[q_fn]
                all_neg_feats = []
                for n_fn in all_neg_fns:
                    all_neg_feats.append(all_feats[n_fn])

                all_neg_feats = torch.cat(all_neg_feats, dim=0)
                similarity = q_feat.reshape(1, -1) @ all_neg_feats.t()
                k = similarity.shape[1]
                _, indexs = torch.topk(similarity, dim=1, largest=True, k=k)
                self.hard_negative_pairs[q_fn] = []
                for i in range(indexs.shape[1]):
                    self.hard_negative_pairs[q_fn].append(all_neg_fns[indexs[0, i]])

            print('Find hardest positives...')
            self.hard_positive_pairs = {}
            for q_fn in tqdm(self.valid_fns, total=len(self.valid_fns)):
                q_feat = all_feats[q_fn]
                all_pos_fns = self.positive_pairs[q_fn]
                all_pos_feats = []
                for p_fn in all_pos_fns:
                    all_pos_feats.append(all_feats[p_fn])
                all_pos_feats = torch.cat(all_pos_feats, dim=0)
                similarity = q_feat.reshape(1, -1) @ all_pos_feats.t()
                k = similarity.shape[0]
                _, indexs = torch.topk(similarity, dim=1, largest=False, k=k)
                self.hard_positive_pairs[q_fn] = []
                for i in range(indexs.shape[1]):
                    self.hard_positive_pairs[q_fn].append(all_pos_fns[indexs[0, i]])

        del all_neg_feats
        del all_pos_feats
        del similarity
        del indexs
        del q_feat
        del all_feats

    def read_fns(self, all_fns):
        all_imgs = []
        if self.cls:
            all_cls = []

        for fn in all_fns:
            if len(self.cats) > 1:
                cat_id = random.randint(1, len(self.cats))
            else:
                cat_id = 1

            if self.imgs is None:
                # print(osp.join(self.image_path, self.cats[cat_id - 1], fn))
                img = cv2.imread(osp.join(self.image_path, self.cats[cat_id - 1], fn))
                if self.keep_ratio:
                    size = img.shape[:2][::-1]
                    w, h = size
                    scale = self.R / max(h, w)
                    h_new, w_new = int(round(h * scale)), int(round(w * scale))
                    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

                    img_pad = np.zeros(shape=(self.R, self.R, 3), dtype=np.uint8)
                    img_pad[0:h_new, 0:w_new, :] = img
                    img = img_pad
                else:
                    img = cv2.resize(img, dsize=(self.R, self.R))
            else:
                img = self.imgs[cat_id - 1][fn]

            if self.transform is not None:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img)
            all_imgs.append(img)
            if self.cls:
                if fn not in self.cls_labels.keys():
                    if self.segs is None:
                        seg_img = cv2.imread(osp.join(self.label_path, fn.replace('jpg', 'png')))
                        seg_img = cv2.resize(seg_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
                    else:
                        seg_img = self.segs[fn]
                    filtered_seg, gids = self.seg_to_gid(seg_img=seg_img, grgb_gid_map=self.grgb_gid)
                    gids = np.asarray(gids, np.int64)
                    gids = torch.from_numpy(gids)
                    gids = torch.LongTensor(gids)

                    cls_label = np.zeros(shape=(self.classes), dtype=np.float)
                    uids = np.unique(gids).tolist()
                    for id in uids:
                        cls_label[id] = 1.0
                    cls_label = torch.from_numpy(cls_label)

                    self.cls_labels[fn] = cls_label
                else:
                    cls_label = self.cls_labels[fn]
                all_cls.append(cls_label)

        output = {
            'imgs': all_imgs,
        }
        if self.cls:
            output['cls'] = all_cls
        return output

    def __getitem__(self, idx):
        fn = self.valid_fns[idx]
        # neg_ids = random.sample(range(len(self.valid_fns)), len(self.positive_pairs[fn]) + 2)

        if self.hard_positive_pairs is not None:
            pos_fns = self.hard_positive_pairs[fn][:self.n_pos]
        else:
            pos_ids = random.sample(range(len(self.positive_pairs[fn])), self.n_pos)
            pos_fns = [self.positive_pairs[fn][i] for i in pos_ids]

        if self.hard_negative_pairs is not None:
            neg_fns = self.hard_negative_pairs[fn][:self.n_neg]
        else:
            neg_ids = random.sample(range(len(self.negative_pairs[fn])), self.n_neg)
            neg_fns = [self.negative_pairs[fn][i] for i in neg_ids]

        pos_output = self.read_fns(all_fns=pos_fns)
        neg_output = self.read_fns(all_fns=neg_fns)
        query_output = self.read_fns(all_fns=[fn])

        output = {
            "query_imgs": query_output['imgs'],
            "pos_imgs": pos_output['imgs'],
            "neg_imgs": neg_output['imgs'],
        }

        if self.cls:
            output['query_cls'] = query_output['cls']
            output['pos_cls'] = pos_output['cls']
            output['neg_cls'] = neg_output['cls']
        return output

    def __len__(self):
        return len(self.valid_fns)
