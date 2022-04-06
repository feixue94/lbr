# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> train_place_recog
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   23/08/2021 21:08
=================================================='''

import torch
import torch.nn as nn
import argparse
import torch.utils.data as Data
from torchvision.models import vgg16
import torchvision.transforms as tvf
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import cv2
from math import ceil
import json
from tools.common import save_args

from net.plceregnets.pregnet import NetVLAD
from net.plceregnets.gem import GEM
from loss.aploss import APLoss
from dataloader.place_recog import PlaceRecog
from localization.coarse.evaluate import evaluate_retrieval, evaluate_retrieval_by_query
from tools.common import torch_set_gpu


def get_retrival_encoder(encoder, input_dim=512, name='resnext101_32x4d', depth=4, weights='ssl', tll=False):
    # encoder = get_encoder(name=name, weights=weights, depth=depth)
    if input_dim == 256:
        encoder_dim = 512
        layers = [
            encoder.conv1,
            encoder.bn1,  # 2x ds
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,  # 4x ds
        ]  # layer1, output_dim=256, ds: 4x
        layers.append(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            )
        )
    elif input_dim == 512:
        encoder_dim = 512
        layers = [
            encoder.conv1,
            encoder.bn1,  # 2x ds
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,  # 4x ds
            encoder.layer2,  # 4x ds
        ]

        layers.append(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            )
        )
    elif input_dim == 1024:
        encoder_dim = 512
        layers = [
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,  # 2x ds
            encoder.layer1,  # 4x ds
            encoder.layer2,  # 8x ds
            encoder.layer3,  # 16x ds
        ]  # layer1,2,3, output_dim=1024, ds: 16x
        # layers.append(
        #     nn.Sequential(
        #         nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        #     )
        # )
    elif input_dim == 2048:
        layers = [
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,  # 2x ds
            encoder.layer1,  # 4x ds
            encoder.layer2,  # 8x ds
            encoder.layer3,  # 16x ds
            encoder.layer4,  # 32x ds
        ]  # layer1,2,3, output_dim=1024, ds: 16x

    if tll:
        dynamic_layer = -1
        for l in layers[:dynamic_layer]:
            for p in l.parameters():
                p.requires_grad = False
    else:
        dynamic_layer = 0
        for l in layers:
            for p in l.parameters():
                p.requires_grad = False

    return nn.Sequential(*layers)


class PlaceRecTrainer:
    def __init__(self, model, encoder, loss_func, train_loader, train_set, args=None):
        self.args = args
        self.model = model
        self.encoder = encoder
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.num_epochs = self.args.epochs
        self.epoch = 0
        self.iteration = 0
        self.init_lr = self.args.lr
        self.weight_decay = self.args.weight_decay
        self.train_set = train_set
        self.cls = args.cls > 0
        self.do_eval = (self.args.do_eval > 0)

        self.eval_imgs = {}

        if self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                params=[p for p in self.model.parameters() if p.requires_grad],
                lr=self.init_lr,
                weight_decay=self.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.init_lr,
                                        weight_decay=self.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=self.init_lr,
                                       weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        now = datetime.datetime.now()
        self.save_dir = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_dir = self.save_dir + "_" + self.args.dataset + "_" + str(self.args.encoder_net) + '_' + str(
            self.args.network) + "_B" + str(self.args.bs) + "_R" + str(
            self.args.R) + '_E' + str(args.epochs) + "_iD" + str(
            self.args.indim) + "_oD" + str(
            self.args.outdim) + "_S" + str(len(args.scales)) + "_" + args.pooling + '_' + args.loss

        if self.args.loss == 'tri':
            self.save_dir = self.save_dir + '_m' + str(args.margin)
        if self.args.loss == 'ap':
            self.save_dir = self.save_dir + '_P' + str(self.args.n_pos) + '_N' + str(self.args.n_neg)

        if self.cls:
            self.save_dir += ('_cls')
        if args.hard > 0:
            self.save_dir += ('_hard')
        if args.tll > 0:
            self.save_dir += ('_tll')
        if args.attention > 0:
            self.save_dir += ('_att')
        if args.do_proj > 0:
            self.save_dir += ("_proj")
        if args.keep_ratio > 0:
            self.save_dir += ("_kr")

        self.tag = self.save_dir
        self.save_dir = osp.join(self.args.save_root, self.save_dir)
        self.writer = SummaryWriter(self.save_dir)
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.log_file = open(osp.join(self.save_dir, "log.txt"), "a+")
        save_args(args=args, save_path=osp.join(self.save_dir, "args.txt"))
        if len(self.args.gpu) > 1:
            device_ids = [i for i in range(len(args.gpu))]
            self.model = nn.DataParallel(self.model, device_ids=device_ids).cuda()
            self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids).cuda()
        else:
            self.model = self.model.cuda()
            self.encoder = self.encoder.cuda()

    def calc_retrieval_loss(self, query_feats, pos_feats, neg_feats):
        if self.args.loss == 'tri':
            loss = self.loss_func(query_feats, pos_feats, neg_feats)
        elif self.args.loss == 'ap':
            all_feats = torch.cat([pos_feats, neg_feats])
            # query_feats_vec = query_feats.view(query_feats.shape[0], -1)
            scores = query_feats @ all_feats.t()
            labels = scores.new_zeros(scores.shape, dtype=torch.uint8)
            # print('qfeat/all_feat', query_feats.shape, pos_feats.shape, neg_feats.shape, all_feats.shape)
            # print(scores.shape, labels.shape)
            labels[:, :pos_feats.shape[0]] = 1
            loss = self.loss_func(scores, labels)

        return loss

    def get_clusters(self, img_dir, db_img_path, centroid_fn):
        self.model.eval()

        val_transform = tvf.Compose(
            (
                tvf.ToPILImage(),
                tvf.Resize((self.args.R, self.args.R)),
                # tvf.Resize((1024, 1024)),
                tvf.ToTensor(),
                tvf.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            )
        )

        db_fns = []
        with open(db_img_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                db_fns.append(l.strip())

        print('Initialize centroids...')
        nDescriptors = 50000
        nPerImage = 100
        nIm = ceil(nDescriptors / nPerImage)

    def process_epoch(self):
        avg_loss = []
        self.model.train()
        for batch_idx, inputs in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
            # if batch_idx >= 10:
            #     break
            self.iteration = self.iteration + 1
            self.optimizer.zero_grad()
            query_imgs = torch.vstack(inputs["query_imgs"]).cuda()
            pos_imgs = torch.vstack(inputs["pos_imgs"]).cuda()
            neg_imgs = torch.vstack(inputs["neg_imgs"]).cuda()
            Bq = query_imgs.shape[0]
            Bp = pos_imgs.shape[0]
            Bn = neg_imgs.shape[0]
            # print("imgs: ", query_imgs.shape, pos_imgs.shape, neg_imgs.shape)
            imgs = torch.vstack([query_imgs, pos_imgs, neg_imgs])

            outputs = self.model(imgs)
            out_feat = outputs['feat']
            # print(feats.shape, out_feat.shape)
            loss = self.calc_retrieval_loss(query_feats=out_feat[:Bq], pos_feats=out_feat[Bq:Bq + Bp],
                                            neg_feats=out_feat[Bq + Bp:])

            if self.cls:
                pred_cls = outputs['cls']
                query_cls = torch.vstack(inputs['query_cls'])
                pos_cls = torch.vstack(inputs['pos_cls'])
                neg_cls = torch.vstack(inputs['neg_cls'])
                gt_cls = torch.vstack([query_cls, pos_cls, neg_cls]).cuda()
                cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_cls, gt_cls)

                loss = loss + cls_loss * args.weight_cls
            else:
                cls_loss = torch.zeros_like(loss)

            loss.backward()
            self.optimizer.step()
            avg_loss.append(loss.item())

            if batch_idx % self.args.log_interval == 0:
                text = "[Train epoch {:d}-batch {:d}/{:d} avg loss {:.4f} cls loss: {:.4f}]\n".format(self.epoch,
                                                                                                      batch_idx,
                                                                                                      len(
                                                                                                          self.train_loader),
                                                                                                      loss.item(),
                                                                                                      cls_loss.item())
                print(text)
                self.log_file.write(text)

                self.writer.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=self.iteration)

        text = "[Train epoch {:d} avg loss {:.4f}]\n".format(self.epoch, np.mean(avg_loss))
        print(text)
        self.log_file.write(text)
        self.log_file.flush()

        del outputs
        del out_feat
        # del feats
        return np.mean(avg_loss)

    def train(self, start_epoch=0):
        max_acc = -1
        acc_history = []
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch

            # if self.args.preload > 0:
            #     self.train_loader.dataset.load_images()

            self.process_epoch()

            # if self.do_eval:
            acc = self.evaluate(img_dir=self.args.img_dir, query_img_path=args.query_img_path,
                                db_img_path=args.db_img_path,
                                gt_fn=args.gt_fn, topk=[1, 5, 10, 20, 50])

            if epoch % 10 == 0 and self.args.hard > 0:
                self.train_set.update_negatives(model=self.model, encoder=self.encoder, dim=args.dim, topk=50,
                                                use_atten=self.args.attention > 0)
            acc_history.append(acc)

            self.scheduler.step()

            checkpoint_path = os.path.join(
                self.save_dir, '{:s}_{:02d}.pth'.format(self.args.network, self.epoch)
            )

            checkpoint = {
                "epoch": self.epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }

            if len(args.gpu) > 1:
                checkpoint["state_dict"] = self.model.module.state_dict()
                checkpoint["model"] = self.model.module
            else:
                checkpoint["state_dict"] = self.model.state_dict()
                checkpoint["model"] = self.model

            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)

            if self.do_eval:
                if acc_history[-1] > max_acc:
                    max_acc = acc_history[-1]
                    best_checkpoint_path = os.path.join(
                        self.save_dir,
                        'best.pth')
                    torch.save(checkpoint, best_checkpoint_path, _use_new_zipfile_serialization=False)
            # shutil.copy(checkpoint_path, best_checkpoint_path)

            self.log_file.flush()

        self.log_file.close()

    def prediction(self, img):
        outputs = self.model(img)
        out_feat = outputs['feat']

        return out_feat

    def evaluate(self, img_dir, query_img_path, db_img_path, gt_fn, topk=[1, 10, 20, 50]):
        print('Start evaluation...')
        self.model.eval()
        val_transform = tvf.Compose(
            (
                tvf.ToPILImage(),
                tvf.ToTensor(),
                tvf.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            )
        )

        all_fn_feats = {}

        query_fns = []
        with open(query_img_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                query_fns.append(l.strip())

        db_fns = []
        with open(db_img_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                db_fns.append(l.strip())

        print('Load {:d} query and {:d} db images'.format(len(query_fns), len(db_fns)))
        # db_fn_feats = {}
        with torch.no_grad():
            all_db_feats = []
            all_query_feats = []
            for fn in tqdm(db_fns, total=len(db_fns)):
                if fn in self.eval_imgs.keys():
                    img = self.eval_imgs[fn]
                else:
                    img = cv2.imread(osp.join(img_dir, fn))
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.args.keep_ratio > 0:
                        size = img.shape[:2][::-1]
                        w, h = size
                        scale = self.args.R / max(h, w)
                        h_new, w_new = int(round(h * scale)), int(round(w * scale))
                        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
                        img_pad = np.zeros(shape=(self.args.R, self.args.R, 3), dtype=np.uint8)
                        img_pad[0:h_new, 0:w_new, :] = img
                        img = img_pad
                    else:
                        img = cv2.resize(img, dsize=(self.args.R, self.args.R))
                    if self.args.preload > 0:
                        self.eval_imgs[fn] = img

                img = val_transform(img).unsqueeze(0).cuda()

                out_feat = self.prediction(img=img)
                all_db_feats.append(out_feat)

                all_fn_feats[fn] = out_feat.squeeze().cpu().numpy()

                # db_fn_feats[fn] = out_feat

                # if len(all_db_feats) >= 50:
                #     break

            for fn in tqdm(query_fns, total=len(query_fns)):
                if fn in self.eval_imgs.keys():
                    img = self.eval_imgs[fn]
                else:
                    img = cv2.imread(osp.join(img_dir, fn))
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.args.keep_ratio > 0:
                        size = img.shape[:2][::-1]
                        w, h = size
                        scale = self.args.R / max(h, w)
                        h_new, w_new = int(round(h * scale)), int(round(w * scale))
                        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
                        img_pad = np.zeros(shape=(self.args.R, self.args.R, 3), dtype=np.uint8)
                        img_pad[0:h_new, 0:w_new, :] = img
                        img = img_pad
                    else:
                        img = cv2.resize(img, dsize=(self.args.R, self.args.R))
                    if self.args.preload > 0:
                        self.eval_imgs[fn] = img
                img = val_transform(img).unsqueeze(0).cuda()
                out_feat = self.prediction(img=img)
                all_query_feats.append(out_feat)

                all_fn_feats[fn] = out_feat.squeeze().cpu().numpy()

                # if len(all_query_feats) >= 20:
                #     break

            all_query_feats = torch.cat(all_query_feats, dim=0)
            all_db_feats = torch.cat(all_db_feats, dim=0)
            sim = all_query_feats @ all_db_feats.t()

            # print(all_db_feats.shape, all_query_feats.shape)

            k = np.max(topk)
            if k > all_db_feats.shape[0]:
                k = all_db_feats.shape[0]

            pred_idxes = torch.topk(sim, dim=1, k=k, largest=True)[1]

            pred_retrievals = {}
            for i in range(pred_idxes.shape[0]):
                q_fn = query_fns[i]
                cans = []
                for j in range(k):
                    cans.append(db_fns[pred_idxes[i, j]])

                pred_retrievals[q_fn] = cans

        gt_retrievals = {}
        with open(gt_fn, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split()
                if l[0] in gt_retrievals.keys():
                    # if len(gt_retrievals[l[0]]) < 5:
                    gt_retrievals[l[0]].append(l[1])
                else:
                    gt_retrievals[l[0]] = [l[1]]

        results = evaluate_retrieval(preds=pred_retrievals, gts=gt_retrievals, ks=topk)
        success_ratio = evaluate_retrieval_by_query(preds=pred_retrievals, gts=gt_retrievals, ks=topk)

        for k in topk:
            text = 'Eval epoch:{:d} top@{:d} acc:{:.2f} recall:{:.2f}, success:{:.2f}' \
                .format(self.epoch, k,
                        results[k]['accuracy'],
                        results[k]['recall'],
                        success_ratio[k])
            print(text)
            self.log_file.write(text + '\n')

        for fn in success_ratio['failed_case']:
            self.log_file.write(fn + '\n')
        self.log_file.write('\n')

        # record retrieval results
        with open(
                osp.join(self.save_dir, 'pairs_epoch{:d}_{:s}_{:d}.txt'.format(self.epoch, self.args.network, k)),
                'w') as f:
            for q in pred_retrievals.keys():
                cans = pred_retrievals[q]
                for c in cans:
                    text = '{:s} {:s}'.format(q, c)
                    f.write(text + '\n')

        np.save(osp.join(self.save_dir, 'feat_epoch{:d}_{:s}.npy'.format(self.epoch, self.args.network)), all_fn_feats)
        del all_query_feats
        del all_db_feats
        del sim
        # return results[5]['recall']

        for k in topk:
            self.writer.add_scalar(tag='success_ratio_{:d}'.format(k), scalar_value=success_ratio[k],
                                   global_step=self.epoch)
        return success_ratio[10]


def main(args):
    torch_augs = [tvf.ToPILImage(), tvf.ToTensor()]
    if 'colorjitter' in args.torch_aug:
        torch_augs.append(tvf.ColorJitter(0.2, 0.2, 0.2, 0.1))
    if 'normalize' in args.torch_aug:
        torch_augs.append(tvf.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]))
    # if 'randomerasing' in args.torch_aug:
    #     torch_augs.append(tvf.RandomErasing(p=0.5, scale=(0.02, 0.5)))

    train_transform = tvf.Compose(torch_augs)
    print(train_transform)

    if args.encoder_net == 'resnet101':
        encoder = torch.load(osp.join(args.root, args.encoder)).encoder  # .eval()
        # encoder = get_encoder(name='resnext101_32x4d', weights='ssl', depth=4)
        # encoder = get_retrival_encoder(encoder=encoder, input_dim=args.dim, tll=args.tll > 0)

        if args.network == 'netvlad':
            # print("encoder: ", list(encoder.children())[:-1])
            # exit(0)
            net = NetVLAD(encoder=encoder,
                          num_clusters=args.n_cluster, dim=512, in_dim=args.dim, projection=args.do_proj > 0,
                          n_classes=args.classes,
                          proj_layer=None,
                          cls=args.cls > 0)
            # centroid_fn = osp.join(args.save_root, "netvlad_aachen_R{:d}_proj_64_desc_cen.hdf5".format(args.R))
            # with h5py.File(centroid_fn, mode='r') as h5:
            #     clsts = h5.get("centroids")[...]
            #     traindescs = h5.get("descriptors")[...]
            #     net.init_params(clsts, traindescs)
            #     del clsts, traindescs
            # if args.dim == 512:
            net.load_state_dict(torch.load("models/netvlad.pth"), strict=False)
        elif args.network == 'gem':
            # for p in encoder.parameters():
            #     p.requires_grad = False

            net = GEM(encoder=encoder,
                      in_dim=args.indim,
                      scales=args.scales,
                      out_dim=args.outdim,
                      projection=args.do_proj > 0,
                      n_classes=args.classes,
                      cls=args.cls > 0,
                      pooling=args.pooling)
    elif args.encoder_net == 'vgg16':
        encoder = vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        if args.tll > 0:
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False
        encoder = nn.Sequential(*layers)
        net = NetVLAD(encoder=encoder,
                      num_clusters=args.n_cluster, dim=512, in_dim=args.dim, projection=args.do_proj > 0,
                      n_classes=args.classes,
                      proj_layer=None,
                      cls=args.cls > 0)
        net.load_state_dict(torch.load("models/netvlad.pth"), strict=False)
        net.load_state_dict(torch.load('weights/vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar')['state_dict'],
                            strict=False)

    print("net: ", net)
    for name, p in net.named_parameters():
        if p.requires_grad:
            print('training with gradients: ', name)
    # print("encoder: ", encoder)

    if args.loss == 'tri':
        loss_func = nn.TripletMarginLoss(margin=args.margin ** 0.5,
                                         p=2, reduction='mean')
    elif args.loss == 'ap':
        loss_func = APLoss().cuda()

    # train_loader = None
    trainset = PlaceRecog(image_path=osp.join(args.root, args.train_image_path),
                          img_list=args.train_imglist,
                          positive_pairs=args.positive_pairs,
                          cats=args.train_cats,
                          train=True,
                          aug=None,
                          R=args.R,
                          n_pos=args.n_pos,
                          n_neg=args.n_neg,
                          transform=train_transform,
                          label_path=osp.join(args.root, args.train_label_path),
                          grgb_gid_file=args.grgb_gid_file,
                          n_classes=args.classes,
                          # cls=True,
                          use_cls=args.cls > 0,
                          preload=args.preload > 0,
                          keep_ratio=args.keep_ratio > 0,
                          )

    train_loader = Data.DataLoader(dataset=trainset,
                                   batch_size=args.bs,
                                   num_workers=args.workers,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=True, )
    #
    # checkpoint = torch.load('/data/cornucopia/fx221/exp/shloc/aachen/2021_08_25_23_46_44_aachen_netvlad_resnext101_32x4d_d4_u8_b16_R256_E60_nC64_nD512_proj/2021_08_25_23_46_44_aachen_netvlad_resnext101_32x4d_d4_u8_b16_R256_E60_nC64_nD512_proj.best.pth')
    # net.load_state_dict(checkpoint['model'], strict=True)
    # checkpoint['model'] = net
    # torch.save(checkpoint, 'netvlad_full.pth',_use_new_zipfile_serialization=False)
    # exit(0)

    trainer = PlaceRecTrainer(model=net, encoder=encoder, train_loader=train_loader, train_set=trainset,
                              loss_func=loss_func, args=args)
    trainer.train(start_epoch=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Semantic localization Network")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--dataset", type=str, default="small", help="small, large, robotcar")
    parser.add_argument("--network", type=str, default="unet")
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--out_channels", type=int, default=512)
    parser.add_argument("--root", type=str, default="/home/mifs/fx221/data/cam_street_view")
    parser.add_argument("--train_label_path", type=str,
                        default="camvid_360_cvpr18_P2_training_data/building_only_filtered1_hand_labels_prop")
    parser.add_argument("--train_image_path", type=str, default="camvid_360_cvpr18_P2_training_data/images_hand")
    parser.add_argument("--val_label_path", type=str,
                        default="camvid_360_cvpr18_P4_testing_data/building_only_filtered1_hand_labels_prop")
    parser.add_argument("--val_image_path", type=str, default="camvid_360_cvpr18_P4_testing_data/images_hand")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--n_pos", type=int, default=1)
    parser.add_argument("--n_neg", type=int, default=1)
    parser.add_argument("--n_cluster", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--R", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hard", type=int, default=0)
    parser.add_argument("--cls", type=int, default=0)
    parser.add_argument("--attention", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    # parser.add_argument("--save_dir", type=str, default=None)
    # parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--aug", dest="aug", action="store_true", default=False)
    parser.add_argument("--preload", dest="preload", action="store_true", default=False)
    parser.add_argument("--ignore_bg", dest="ignore_bg", action="store_true", default=False)
    parser.add_argument("--weight_cls", type=float, default=1.0)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default='timm-resnest50d')
    parser.add_argument("--encoder_weights", type=str, default='imagenet')
    parser.add_argument("--save_root", type=str, default="/home/mifs/fx221/fx221/exp/shloc/aachen")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')
    parser.add_argument("--milestones", type=list, default=[60, 80])
    parser.add_argument("--grgb_gid_file", type=str)
    parser.add_argument("--train_imglist", type=str)
    parser.add_argument("--test_imglist", type=str)
    parser.add_argument("--train_cats", type=list, default=None)
    parser.add_argument("--val_cats", type=list, default=None)
    parser.add_argument("--positive_pairs", type=str, default=None)
    parser.add_argument("--classes", type=int, default=400)
    parser.add_argument("--img_dir", type=str,
                        default='/data/cornucopia/fx221/localization/aachen_v1_1/images/images_upright')
    parser.add_argument("--query_img_path", type=str,
                        default='/home/mifs/fx221/Research/Code/shloc/datasets/aachen/aachen_query_imglist.txt')
    parser.add_argument("--db_img_path", type=str,
                        default='/home/mifs/fx221/Research/Code/shloc/datasets/aachen/aachen_db_imglist.txt')
    parser.add_argument("--gt_fn", type=str,
                        default='/home/mifs/fx221/Research/Code/shloc/datasets/aachen/pairs-query-aachen-gt-netvlad.txt')
    # default='/home/mifs/fx221/Research/Code/Hierarchical-Localization/pairs/aachen_v1.1/pairs-query-netvlad50.txt')

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    torch_set_gpu(gpus=args.gpu)
    # print(args)
    # exit(0)
    main(args=args)
