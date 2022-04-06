# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> trainer_recog
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   29/04/2021 20:21
=================================================='''
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
from tools.common import save_args
import cv2
from tools.seg_tools import seg_to_rgb, label_accuracy_score, label_to_rgb, rgb_to_bgr
from loss.seg_loss.crossentropy_loss import cross_entropy_seg, cross_entropy2d, CrossEntropyLossWithOHEM, \
    SoftCrossEntropyLossWithOHEM, FocalLoss
from loss.accuracy import accuracy

from tools.optim import PolyLR


class RecogTrainer:
    def __init__(self, model, loss_func, train_loader, eval_loader, map=None, args=None):
        self.args = args
        self.model = model
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.map = map
        self.seg = args.segmentation
        self.cls = args.classification
        self.hira = args.hierarchical

        self.num_epochs = self.args.epochs
        self.epoch = 0
        self.iteration = 0
        self.init_lr = self.args.lr
        self.weight_decay = self.args.weight_decay
        self.multi_lr = args.multi_lr > 0

        if self.multi_lr:
            params = [
                {'params': self.model.encoder.parameters(), 'lr': 0.1 * self.init_lr},
                {'params': self.model.decoder.parameters(), 'lr': self.init_lr},
                {'params': self.model.seghead.parameters(), 'lr': self.init_lr},
                {'params': self.model.cls_head.parameters(), 'lr': self.init_lr},
            ]
        else:
            params = [
                {'params': self.model.parameters(), 'lr': self.init_lr},
            ]

        if self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                params=params,
                lr=self.init_lr,
                weight_decay=self.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(params=params, lr=self.init_lr,
                                        weight_decay=self.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(params, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)

        # self.schduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
        #                                                      mode='min', factor=0.5, patience=10)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
        #                                                milestones=[80, 120]) # 60, 80
        # self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
        #                                                 milestones=args.milestones)  # 60, 80

        if self.args.seg_loss == 'ce':
            self.seg_loss_func = torch.nn.CrossEntropyLoss().cuda()
        elif self.args.seg_loss == 'ceohem':
            self.seg_loss_func = CrossEntropyLossWithOHEM(ohem_ratio=0.7).cuda()
        elif self.args.seg_loss == 'sceohem':
            self.seg_loss_func = SoftCrossEntropyLossWithOHEM(ohem_ratio=0.7).cuda()
        elif self.args.seg_loss == 'focal':
            self.seg_loss_func = FocalLoss(size_average=True).cuda()

        if len(args.gpu) == 1:
            self.model = self.model.cuda()
        else:
            device_ids = [i for i in range(len(args.gpu))]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids).cuda()
            # self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=device_ids)
            # self.seg_loss_func = torch.nn.DataParallel(self.seg_loss_func, device_ids=device_ids).cuda()
            # model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

        if args.lr_policy == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                            milestones=args.milestones, gamma=0.1)  # 60, 80
            # self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
            #                                            step_size=20,
            #                                            gamma=0.1)
        elif args.lr_policy == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                  mode='min',
                                                                  factor=0.5,
                                                                  patience=10)
        elif args.lr_policy == 'poly':
            self.scheduler = PolyLR(optimizer=self.optimizer,
                                    # max_iters=self.args.epochs * len(train_loader),
                                    max_decay_steps=self.args.epochs * len(train_loader),
                                    end_learning_rate=1e-6,
                                    power=0.9)

        now = datetime.datetime.now()
        self.save_dir = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_dir = self.save_dir + "_" + self.args.dataset + "_" + str(
            self.args.network) + "_" + self.args.encoder_name + "_d" + str(self.args.encoder_depth) + "_u" + str(
            self.args.upsampling) + "_b" + str(self.args.bs) + "_R" + str(
            self.args.R) + '_E' + str(args.epochs) + '_' + args.seg_loss + '_' + args.optimizer + '_' + args.lr_policy

        if self.multi_lr:
            self.save_dir = self.save_dir + '_mlr'

        if self.seg:
            self.save_dir = self.save_dir + "_seg"

        if self.cls:
            self.save_dir = self.save_dir + "_cls"

        if self.args.aug:
            self.save_dir += '_aug'

        if args.tag is not None:
            self.save_dir += ("_" + args.tag)

        self.tag = self.save_dir
        self.save_dir = osp.join(self.args.save_root, self.save_dir)

        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.writer = SummaryWriter(self.save_dir)

        self.log_file = open(osp.join(self.save_dir, "log.txt"), "a+")
        save_args(args=args, save_path=osp.join(self.save_dir, "args.txt"))

    def compute_seg_loss(self, pred_segs, gt_segs, weights=[1.0, 1.0, 1.0, 1.0]):
        # pred_segs = inputs["masks"]
        # gt_segs = outputs["label"]

        seg_loss = 0

        for pseg, gseg in zip(pred_segs, gt_segs):
            # print("pseg, gseg: ", pseg.shape, gseg.shape)
            gseg = gseg.cuda()
            if len(gseg.shape) == 3:
                gseg = gseg.unsqueeze(1)
            if pseg.shape[2] != gseg.shape[2] or pseg.shape[3] != gseg.shape[3]:
                gseg = F.interpolate(gseg.float(), size=(pseg.shape[2], pseg.shape[3]), mode="nearest")

            # seg_loss += cross_entropy_seg(input=pseg, target=gseg)
            # idx = gseg.long()
            # target = torch.zeros_like(pseg)
            # target = target.scatter_(1, idx, 1).long()
            seg_loss += self.seg_loss_func(pseg, gseg.long().squeeze())

        return seg_loss / len(pred_segs)

    def compute_cls_loss(self, pred_cls, gt_cls, method="cel"):
        cls_loss = 0
        for pc, gc in zip(pred_cls, gt_cls):
            # print("pc, gc: ", pc.shape, gc.shape)
            gc = gc.cuda()
            if method == "cel":
                cls_loss += torch.nn.functional.binary_cross_entropy_with_logits(pc, gc)
            else:
                cls_loss += torch.nn.functional.cross_entropy(pc, torch.max(gc, 1)[1])
        return cls_loss

    def process_epoch(self):
        metrics = []
        # correct = 0
        # total = 0
        save_ids = [0, 100, 200]
        self.model.train()
        for batch_idx, inputs in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
            # if batch_idx >= 10:
            #     break
            self.iteration = self.iteration + 1
            self.optimizer.zero_grad()

            imgs = inputs["img"].cuda()
            outputs = self.model(imgs)

            if type(outputs) == dict:
                loss_item = {}
                total_loss = 0
                if self.seg:
                    if "seg_loss" in outputs.keys():
                        seg_loss = torch.mean(outputs["seg_loss"])
                    else:
                        seg_loss = self.compute_seg_loss(pred_segs=outputs["masks"], gt_segs=inputs["label"])
                    loss_item["seg_loss"] = seg_loss
                    total_loss = total_loss + seg_loss

                    # print('seg_loss: ', seg_loss)

                    pred_seg = outputs["masks"][0]
                    gt_seg = inputs["label"][0].cuda().unsqueeze(1)
                    gt_seg = F.interpolate(gt_seg.float(), size=(pred_seg.shape[2], pred_seg.shape[3]), mode="nearest")
                    gt_seg = gt_seg.long()
                if self.cls:
                    if self.seg:
                        if "cls_loss" in outputs.keys():
                            cls_loss = torch.mean(outputs["cls_loss"])
                        else:
                            cls_loss = self.compute_cls_loss(pred_cls=outputs["cls"], gt_cls=inputs["cls"],
                                                             method="cel")
                    if not self.seg:
                        cls_loss = self.compute_cls_loss(pred_cls=outputs["cls"], gt_cls=inputs["cls"], method="ce")
                    loss_item["cls_loss"] = cls_loss
                    total_loss = total_loss + cls_loss

                    # print('cls_loss: ', cls_loss)

                    _, predicted = (outputs["cls"][0]).max(1)
                    total = inputs["cls"][0].size(0)
                    # print("pred: ", predicted.shape, inputs["cls"][0].shape, outputs["cls"][0].shape)
                    correct = predicted.eq(torch.max(inputs["cls"][0].cuda(), 1)[1]).sum().item()
                loss_item["loss"] = total_loss

            else:
                loss_inputs = {}
                if self.seg:
                    pred_seg = outputs[0]
                    # print("pred_seg: ", pred_seg.shape)
                    gt_seg = inputs["label"].cuda().unsqueeze(1)  # to(self.model.device)
                    # gt_seg = inputs["label"].to(self.args.gpu) # to(self.model.device)
                    if pred_seg.shape[2] != gt_seg.shape[2] or pred_seg.shape[3] != gt_seg.shape[3]:
                        gt_seg = F.interpolate(gt_seg.float(),
                                               size=(pred_seg.shape[2], pred_seg.shape[3]),
                                               mode="nearest")
                    # gt_seg = torch.nn.DataParallel(inputs["gt_seg"]).cuda()
                    loss_inputs["pred_seg"] = pred_seg
                    loss_inputs["gt_seg"] = gt_seg
                if self.cls:
                    pred_cls = outputs[1]
                    gt_cls = inputs["cls"].cuda()  # to(self.model.device)
                    # gt_cls = torch.nn.DataParallel(inputs["gt_cls"]).cuda()

                    loss_inputs["pred_cls"] = pred_cls
                    loss_inputs["gt_cls"] = gt_cls
                if self.hira:
                    pred_hiera = outputs[2]
                    gt_hiera = inputs["hiera"].cuda()
                    loss_inputs["pred_hiera"] = pred_hiera
                    loss_inputs["gt_hiera"] = gt_hiera

                loss_item = self.loss_func(**loss_inputs)
            loss = loss_item["loss"]
            if "seg_loss" in loss_item.keys():
                seg_loss = loss_item["seg_loss"]
            else:
                seg_loss = torch.zeros_like(loss)

            if "cls_loss" in loss_item.keys():
                cls_loss = loss_item["cls_loss"]
            else:
                cls_loss = torch.zeros_like(loss)

            if "hiera_loss" in loss_item.keys():
                hiera_loss = loss_item["hiera_loss"]
            else:
                hiera_loss = torch.zeros_like(loss)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.seg:
                pred_labels = pred_seg.max(1)[1].cpu().numpy()
                gt_labels = gt_seg.squeeze().cpu().numpy()
                # acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                #     label_preds=pred_labels,
                #     label_trues=gt_labels,
                #     n_class=self.args.classes,
                # )
                # acc, acc_cls, mean_iu, fwavacc = 0., 0., 0., 0.
                # print(pred_seg.shape, gt_labels.shape)
                acc, acc_cls, fwavacc = accuracy(pred=pred_seg, target=gt_seg.squeeze(), topk=(1, 5, 10))
                acc = acc.cpu().data.numpy()
                fwavacc = fwavacc.cpu().data.numpy()
                acc_cls = acc_cls.cpu().data.numpy()
                mean_iu = 0
            else:
                acc, acc_cls, mean_iu, fwavacc = 0., 0., 0., 0.
                acc = correct / total

            metrics.append(
                [loss.item(), seg_loss.item(), cls_loss.item(), hiera_loss.item(), acc, acc_cls,
                 mean_iu, fwavacc])
            if batch_idx % self.args.log_interval == 0:
                mean_metrics = np.mean(np.array(metrics, dtype=np.float), axis=0)
                # print("mean_metrics: ", mean_metrics.shape)

                text = '[Train epoch {:d}-batch {:d}/{:d} | avg-loss:{:.3f} seg:{:.3f} cls:{:.3f} acc:{:.3f} ' \
                       'acc_cls:{:.3f} iu:{:.3f} fwavacc:{:.3f}]\n'.format(
                    self.epoch, batch_idx, len(self.train_loader), mean_metrics[0],
                    mean_metrics[1], mean_metrics[2], mean_metrics[3], mean_metrics[4],
                    mean_metrics[5], mean_metrics[6])
                self.log_file.write(text + "\n")

                print(text)

                infos = {
                    "loss": loss.item(),
                    "seg_loss": seg_loss.item(),
                    "cls_loss": cls_loss.item(),
                    "hiera_loss": hiera_loss.item(),
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "acc": acc,
                    "acc_cls": acc_cls,
                    "mean_iu": mean_iu,
                    "fwavacc": fwavacc
                }
                for tag, value in infos.items():
                    self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.iteration + 1)

            if self.seg and batch_idx in save_ids:
                raw_img = imgs.cpu().numpy()[0]
                # raw_img = imgs.numpy()[0]
                raw_img = np.transpose(raw_img, (1, 2, 0))
                raw_img = np.uint8(((raw_img + 1.0) * 128))
                pred_label = pred_labels[0]
                gt_label = gt_labels[0]

                if self.hira:
                    pred_hiera = int(pred_hiera.max(1)[1].cpu().numpy()[0])
                    gt_hiera = int(gt_hiera.max(1)[1].cpu().numpy()[0])

                    # print ("pred_hira: ", pred_hiera)
                    pred_seg_img = label_to_rgb(label=pred_label, maps=self.map[pred_hiera])  # RGB
                    gt_seg_img = label_to_rgb(label=gt_label, maps=self.map[gt_hiera])  # RGB
                else:
                    pred_seg_img = label_to_rgb(label=pred_label, maps=self.map)  # RGB
                    gt_seg_img = label_to_rgb(label=gt_label, maps=self.map)  # RGB

                pred_seg_img = rgb_to_bgr(img=pred_seg_img)
                gt_seg_img = rgb_to_bgr(img=gt_seg_img)
                pred_seg_img = cv2.resize(pred_seg_img, dsize=(raw_img.shape[1], raw_img.shape[0]))
                gt_seg_img = cv2.resize(gt_seg_img, dsize=(raw_img.shape[1], raw_img.shape[0]))
                cat_img = np.hstack([raw_img, pred_seg_img, gt_seg_img])
                img_dir = osp.join(self.save_dir, "train-imgs")
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)
                cv2.imwrite(osp.join(img_dir, "epoch-{:d}-{:d}.png".format(self.epoch, batch_idx)), cat_img)

        mean_metrics = np.mean(np.array(metrics, dtype=np.float), axis=0)

        text = "[Train epoch {:d} avg-loss:{:.3f} seg:{:.3f} cls:{:.3f} hiera:{:.3f} acc:{:.3f}  acc_cls:{:.3f} iu:{:.3f} fwavacc:{:.3f}]". \
            format(self.epoch, mean_metrics[0], mean_metrics[1], mean_metrics[2],
                   mean_metrics[3], mean_metrics[4], mean_metrics[5], mean_metrics[6], mean_metrics[7])
        self.log_file.write(text + "\n")
        self.log_file.flush()

        print(text)
        del imgs
        del outputs
        del gt_seg
        return mean_metrics[0]

    def todevice(self, x):
        if isinstance(x, dict):
            return {k: self.todevice(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [self.todevice(v) for v in x]

        if self.iscuda():
            return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()

    def iscuda(self):
        return next(self.model.parameters()).device != torch.device('cpu')

    def evaluate_seg_cls(self):
        metrics = []
        save_ids = [i for i in range(len(self.eval_loader))]
        cls_total = 0
        cls_correct = 0

        self.model.eval()
        for batch_id, inputs in enumerate(tqdm(self.eval_loader, total=len(self.eval_loader))):
            # if batch_id > 10:
            #     break
            with torch.no_grad():
                imgs = inputs["img"].cuda()
                outputs = self.model(imgs)
                if self.seg:
                    pred_seg = outputs["masks"][0]
                    # loss = self.compute_seg_loss(pred_segs=[pred_seg], gt_segs=[inputs["label"][0]])
                    gt_seg = inputs["label"][0].cuda().unsqueeze(1)
                    gt_seg = F.interpolate(gt_seg.float(), size=(pred_seg.shape[2], pred_seg.shape[3]),
                                           mode="nearest")
                    gt_seg = gt_seg.long()
                    pred_labels = pred_seg.max(1)[1].cpu().numpy()
                    gt_labels = gt_seg.squeeze().cpu().numpy()
                    # acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                    #     label_preds=pred_labels,
                    #     label_trues=gt_labels,
                    #     n_class=self.args.classes,
                    # )
                    # acc, acc_cls, mean_iu, fwavacc = loss.item(), 0, 0, 0
                    acc, acc_cls, fwavacc = accuracy(pred=pred_seg, target=gt_seg.squeeze(), topk=(1, 5, 10))
                    acc = acc.cpu().data.numpy()
                    acc_cls = acc_cls.cpu().data.numpy()
                    fwavacc = fwavacc.cpu().data.numpy()
                    mean_iu = 0
                    # acc_cls, mean_iu, fwavacc = 0, 0, 0
                if self.cls:
                    _, predicted = outputs["cls"][0].max(1)
                    cls_total += inputs["cls"][0].size(0)
                    # print("pred: ", predicted.shape, inputs["cls"][0].shape, outputs["cls"][0].shape)
                    cls_correct += predicted.eq(torch.max(inputs["cls"][0].cuda(), 1)[1]).sum().item()

                    batch_cls_correct = predicted.eq(torch.max(inputs["cls"][0].cuda(), 1)[1]).sum().item()
                    batch_cls_total = inputs["cls"][0].size(0)

                metrics.append(
                    [acc, acc_cls, mean_iu, fwavacc, batch_cls_correct / batch_cls_total])

                if self.seg and batch_id in save_ids:
                    raw_img = imgs.cpu().numpy()[0]
                    raw_img = np.transpose(raw_img, (1, 2, 0))
                    raw_img = np.uint8(((raw_img + 1.0) * 128))
                    pred_label = pred_labels[0]
                    gt_label = gt_labels[0]

                    pred_seg_img = label_to_rgb(label=pred_label, maps=self.map)  # RGB
                    gt_seg_img = label_to_rgb(label=gt_label, maps=self.map)  # RGB

                    pred_seg_img = rgb_to_bgr(img=pred_seg_img)
                    gt_seg_img = rgb_to_bgr(img=gt_seg_img)
                    pred_seg_img = cv2.resize(pred_seg_img, dsize=(raw_img.shape[1], raw_img.shape[0]))
                    gt_seg_img = cv2.resize(gt_seg_img, dsize=(raw_img.shape[1], raw_img.shape[0]))
                    cat_img = np.hstack([raw_img, pred_seg_img, gt_seg_img])
                    img_dir = osp.join(self.save_dir, "imgs")
                    if not os.path.exists(img_dir):
                        os.mkdir(img_dir)
                    cv2.imwrite(osp.join(img_dir, "epoch-{:d}-{:d}.png".format(self.epoch, batch_id)), cat_img)

                del imgs
                del outputs
                del gt_seg

        mean_metrics = np.mean(np.array(metrics, np.float), axis=0)
        infos = {
            "eval_acc": mean_metrics[0],
            "eval_acc_cls": mean_metrics[1],
            "eval_mean_iu": mean_metrics[2],
            "eval_fwavacc": mean_metrics[3],
            "eval_cls_acc": mean_metrics[4]
        }
        for tag, value in infos.items():
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.iteration + 1)

        text = '[Eval epoch {:d} avg acc:{:.3f} acc_cls:{:.3f} mean_iu:{:.3f} fwavacc:{:.3f} cls_acc: {:3f}]\n'.format(
            self.epoch, mean_metrics[0], mean_metrics[1], mean_metrics[2], mean_metrics[3],
            cls_correct / cls_total, )
        print(text)
        self.log_file.write(text)
        self.log_file.flush()

        return -mean_metrics[0]

    def resume(self, checkpoint):
        data = torch.load(checkpoint)
        self.model.load_state_dict(data["model"])
        start_epoch = data["epoch"]
        self.scheduler.load_state_dict(data["scheduler"])

        for i in range(0, start_epoch):
            self.scheduler.step()
        self.train(start_epoch=start_epoch + 1)

    def train(self, start_epoch=0):
        min_loss = 1e10
        loss_history = []
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch
            train_loss = self.process_epoch()
            if self.eval_loader is not None:
                eval_loss = self.evaluate_seg_cls()
            else:
                eval_loss = train_loss
            loss_history.append(eval_loss)

            # self.scheduler.step()
            # self.scheduler.step(eval_loss)
            # checkpoint_path = osp.join(self.save_dir, "%s.%02d.pth" % (self.args.network, self.epoch))
            if len(self.args.gpu) > 1:
                checkpoint = {
                    "epoch": self.epoch,
                    "model": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()
                }
            else:
                checkpoint = {
                    "epoch": self.epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()
                }

            # torch.save(checkpoint, checkpoint_path)

            if loss_history[-1] < min_loss:
                min_loss = loss_history[-1]
                best_checkpoint_path = os.path.join(
                    self.save_dir,
                    '%s.best.pth' % (self.tag)
                )
                torch.save(checkpoint, best_checkpoint_path, _use_new_zipfile_serialization=False)
            # shutil.copy(checkpoint_path, best_checkpoint_path)

        self.log_file.close()
