# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> inference
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   03/05/2021 14:49
=================================================='''

import torch
import torchvision.transforms.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from net.netvlad import NetVLAD
import os
import os.path as osp
import tqdm
import cv2
import argparse


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def inference(img, transform=None):
    if transform is not None:
        img = torch.FloatTensor(transform(img))
        # print(torch.mean(img), img.shape)

    if len(img.size()) == 3:
        img = img.unsqueeze(0).cuda()

    with torch.no_grad():
        img_encoding = model.encoder(img)
        vlad_encoding, soft_assign = model.pool(img_encoding)
        # print(vlad_encoding.shape)
        # exit(0)

    return img_encoding, vlad_encoding, soft_assign


def retrieval(query_imglist, db_imglist, topK=50):
    print("load {:d} query images from {:s}\n".format(len(query_imglist), query_imgdir))
    print("load {:d} db images from {:s}\n".format(len(db_imglist), db_imgdir))

    query_feats = []
    db_feats = []
    for q_fn in tqdm.tqdm(query_imglist, total=len(query_imglist)):
        q_img = cv2.imread(osp.join(query_imgdir, q_fn))
        img_encoding, vladfeat, soft_assign = inference(img=q_img, transform=input_transform())
        query_feats.append(vladfeat)

    for db_fn in tqdm.tqdm(db_imglist, total=len(db_imglist)):
        db_img = cv2.imread(osp.join(db_imgdir, db_fn))
        img_encoding, vladfeat, soft_assign = inference(img=db_img, transform=input_transform())
        db_feats.append(vladfeat)

    query_feats = torch.cat(query_feats, dim=0)
    db_feats = torch.cat(db_feats, dim=0)

    distance = torch.matmul(query_feats, db_feats.t())
    _, topk = torch.topk(distance, dim=1, largest=True, k=min([topK, len(db_imglist)]))
    return topk


def get_netvlad():
    encoder = models.vgg16(pretrained=True)
    layers = list(encoder.features.children())[:-2]
    for l in layers[:-5]:
        for p in l.parameters():
            p.requires_grad = False

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module("encoder", encoder)

    netvlad = NetVLAD(num_clusters=64, dim=512, vladv2=False)
    model.add_module("pool", netvlad)
    print("model: ", model)
    model.load_state_dict(torch.load("weights/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar")["state_dict"])
    model = model.cuda().eval()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("--root", type=str, required=True, help="root dir of images")
    parser.add_argument("--save_fn", type=str, required=True, help="save file name")
    args = parser.parse_args()

    transform_fun = input_transform()
    model = get_netvlad()

    root_dir = args.root
    save_fn = args.save_fn
    query_imgdir = osp.join(root_dir, "camvid_360_cvpr18_P2_training_data", "images_hand")
    query_imglist = sorted(os.listdir(query_imgdir))

    db_imgdir = osp.join(root_dir, "cambridge_center_medium", "images")
    db_imglist = []  # sorted(os.listdir(db_imgdir))

    with open("localization/P2_training_hand_cambridge_center_medium_gps_10.txt", "r") as f:
        lines = f.readlines()
        for p in lines:
            p = p.strip("\n").split(" ")
            if p[1] in db_imglist:
                continue
            else:
                db_imglist.append(p[1])

    db_imglist = sorted(db_imglist)

    model.eval()
    results = retrieval(query_imglist=query_imglist, db_imglist=db_imglist, topK=50)

    with open(save_fn, "w") as f:
        for i in range(results.shape[0]):
            for j in range(min(results.shape[1], 50)):
                text = "{:s} {:s}\n".format(query_imglist[i], db_imglist[results[i, j]])
                f.write(text)
