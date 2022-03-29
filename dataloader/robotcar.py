# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> robotcar
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   18/07/2021 10:22
=================================================='''
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import *
import numpy as np
import os
import os.path as osp
from pathlib import Path
from glob import glob
import tqdm
import random
import time
import cv2
import copy
from threading import Thread
from dataloader.augmentation import RandomRotation, RandomHorizontalFlip, RandomSizedCrop
import torchvision.transforms as tvf
import random
from tools.common import resize_img



class RobotCarSegFull(data.Dataset):
    def __init__(self,
                 image_path,
                 label_path,
                 grgb_gid_file,
                 n_classes,
                 cats,
                 train=True,
                 transform=None,
                 use_cls=False,
                 img_list=None,
                 preload=False,
                 aug=None,
                 R=256,
                 keep_ratio=False,
                 ):
        super(RobotCarSegFull, self).__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.train = train
        self.cls = use_cls
        self.aug = aug
        self.cats = cats
        self.R = R
        print("augmentation: ", self.aug)
        self.classes = n_classes
        self.keep_ratio = keep_ratio

        self.valid_fns = []
        with open(img_list, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                if not osp.exists(osp.join(self.label_path, l.replace("jpg", "png"))):
                    continue

                keep = True
                for c in self.cats:
                    if not osp.exists(osp.join(self.image_path, c, l)):
                        keep = False
                if not keep:
                    continue
                self.valid_fns.append(l)

        self.grgb_gid = {}
        with open(grgb_gid_file, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split(" ")
                self.grgb_gid[int(l[0])] = int(l[1])

        print("Load {:d} valid samples.".format(len(self.valid_fns)))
        print("No. of gids: {:d}".format(len(self.grgb_gid.keys())))

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

    def get_item_seg(self, idx):
        fn = self.valid_fns[idx]

        cat_id = random.randint(1, len(self.cats))
        # print(osp.join(self.image_path, self.cats[cat_id - 1], fn))
        img = cv2.imread(osp.join(self.image_path, self.cats[cat_id - 1], fn))
        # raw_img = cv2.resize(img, dsize=(self.R, self.R))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg_img = cv2.imread(osp.join(self.label_path, fn.replace('jpg', 'png')))

        if self.aug is not None:
            for a in self.aug:
                img, seg_img = a((img, seg_img))

        else:
            if self.keep_ratio:
                img = resize_img(img=img, nh=self.R, mode=cv2.INTER_NEAREST)
            else:
                img = cv2.resize(img.astype(np.uint8), dsize=(self.R, self.R))
            seg_img = cv2.resize(seg_img.astype(np.uint8), dsize=(img.shape[1], img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        raw_img = img.astype(np.uint8)

        if self.transform is not None:
            img = self.transform(img)
        seg_img = np.array(seg_img)

        filtered_seg, gids = self.seg_to_gid(seg_img=seg_img, grgb_gid_map=self.grgb_gid)
        gids = np.asarray(gids, np.int64)
        gids = torch.from_numpy(gids)
        gids = torch.LongTensor(gids)

        output = {
            "raw_img": raw_img,
            "img": img,
            "label": [gids],
            "label_img": filtered_seg,
        }

        if self.cls:
            cls_label = np.zeros(shape=(self.classes), dtype=np.float)
            uids = np.unique(gids).tolist()
            for id in uids:
                cls_label[id] = 1.0
            cls_label = torch.from_numpy(cls_label)

            output["cls"] = [cls_label]

        return output

    def __getitem__(self, idx):
        return self.get_item_seg(idx=idx)

    def __len__(self):
        return len(self.valid_fns)
