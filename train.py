# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> train
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   20/07/2021 10:41
=================================================='''
import argparse
import json
from tools.seg_tools import read_seg_map_without_group
from dataloader.robotcar import RobotCarSegFull
from dataloader.aachen import AachenSegFull
import os.path as osp
import torch
import torch.utils.data as Data
from dataloader.augmentation import ToPILImage, RandomRotation, \
    RandomSizedCrop, RandomHorizontalFlip, \
    ToNumpy
from net.segnet import get_segnet
from loss.seg_loss.segloss import SegLoss
from trainer_recog import RecogTrainer
from tools.common import torch_set_gpu

import torchvision.transforms as tvf


def get_train_val_loader(args):
    train_transform = tvf.Compose(
        (
            tvf.ToTensor(),
            # tvf.ColorJitter(0.25, 0.25, 0.25, 0.15),
            tvf.ColorJitter(0.25, 0.25, 0.25, 0.15),
            tvf.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        )
    )
    val_transform = tvf.Compose(
        (
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        )
    )

    if args.dataset == "robotcar":
        grgb_gid_file = "./datasets/robotcar/robotcar_rear_grgb_gid.txt"
        map_gid_rgb = read_seg_map_without_group(grgb_gid_file)
        train_imglist = "./datasets/robotcar/robotcar_rear_train_file_list.txt"
        test_imglist = "./datasets/robotcar/robotcar_rear_test_file_list.txt"

        if args.aug:
            aug = [
                ToPILImage(),
                RandomRotation(degree=30),
                RandomSizedCrop(size=256),
                RandomHorizontalFlip(),
                ToNumpy(),
            ]
        else:
            aug = None
        trainset = RobotCarSegFull(image_path=osp.join(args.root, args.train_image_path),
                                   label_path=osp.join(args.root, args.train_label_path),
                                   n_classes=args.classes,
                                   transform=train_transform,
                                   grgb_gid_file=grgb_gid_file,
                                   use_cls=True,
                                   img_list=train_imglist,
                                   preload=False,
                                   aug=aug,
                                   train=True,
                                   cats=["overcast-reference", "night", "night-rain",
                                         "dusk", "dawn", "overcast-summer", "overcast-winter", "sun"]
                                   )
        if args.val > 0:
            valset = RobotCarSegFull(image_path=osp.join(args.root, args.train_image_path),
                                     label_path=osp.join(args.root, args.train_label_path),
                                     cats=["overcast-reference", "night", "night-rain",
                                           "dusk", "dawn", "overcast-summer", "overcast-winter", "sun"],
                                     n_classes=args.classes,
                                     transform=train_transform,
                                     grgb_gid_file=grgb_gid_file,
                                     use_cls=True,
                                     img_list=test_imglist,
                                     preload=False,
                                     train=False)
    elif args.dataset == "aachen":
        # grgb_gid_file = "./datasets/aachen/aachen_grgb_gid.txt"
        # train_imglist = "./datasets/aachen/aachen_train_file_list.txt"
        # test_imglist = "./datasets/aachen/aachen_test_file_list.txt"
        grgb_gid_file = args.grgb_gid_file
        train_imglist = args.train_imglist
        test_imglist = args.test_imglist
        map_gid_rgb = read_seg_map_without_group(grgb_gid_file)

        if args.aug:
            aug = [
                # RandomGaussianBlur(),   # worse results, don't do it?
                ToPILImage(),
                # Resize(size=512),
                # RandomScale(low=0.5, high=2.0),
                # RandomCrop(size=256),
                RandomSizedCrop(size=args.R),
                RandomRotation(degree=45),
                RandomHorizontalFlip(),
                ToNumpy(),
            ]
        else:
            aug = None
        trainset = AachenSegFull(image_path=osp.join(args.root, args.train_image_path),
                                 label_path=osp.join(args.root, args.train_label_path),
                                 n_classes=args.classes,
                                 transform=train_transform,
                                 grgb_gid_file=grgb_gid_file,
                                 use_cls=True,
                                 img_list=train_imglist,
                                 preload=False,
                                 aug=aug,
                                 train=True,
                                 cats=args.train_cats,
                                 )
        if args.val > 0:
            valset = AachenSegFull(image_path=osp.join(args.root, args.train_image_path),
                                   label_path=osp.join(args.root, args.train_label_path),
                                   # cats=["images_aug/images_upright"],
                                   # cats=args.val_cats,
                                   n_classes=args.classes,
                                   transform=val_transform,
                                   grgb_gid_file=grgb_gid_file,
                                   use_cls=True,
                                   img_list=test_imglist,
                                   preload=False,
                                   cats=args.val_cats,
                                   train=False)
    train_loader = Data.DataLoader(dataset=trainset,
                                   batch_size=args.bs,
                                   num_workers=args.workers,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=True,
                                   )
    if args.val:
        val_loader = Data.DataLoader(
            dataset=valset,
            batch_size=8,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
    else:
        val_loader = None

    return train_loader, val_loader, map_gid_rgb


def main(args):
    model = get_segnet(network=args.network,
                       n_classes=args.classes,
                       encoder_name=args.encoder_name,
                       encoder_weights=args.encoder_weights,
                       encoder_depth=args.encoder_depth,
                       upsampling=args.upsampling,
                       out_channels=args.out_channels,
                       classification=args.classification,
                       segmentation=args.segmentation, )
    print(model)
    label_weights = torch.ones([args.classes]).cuda()
    label_weights[0] = 0.5
    loss_func = SegLoss(
        segloss_name=args.seg_loss,
        use_cls=True,
        use_seg=args.segmentation > 0,
        cls_weight=args.weight_cls,
        use_hiera=False,
        hiera_weight=0,
        label_weights=label_weights).cuda()

    train_loader, val_loader, map_gid_rgb = get_train_val_loader(args=args, tag=args.tag)
    trainer = RecogTrainer(model=model, train_loader=train_loader, eval_loader=val_loader if args.val else None,
                           loss_func=loss_func, args=args, map=map_gid_rgb)

    if args.resume is not None:
        trainer.resume(checkpoint=args.resume)
    else:
        trainer.train(start_epoch=0)

        print("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Semantic localization Network")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--dataset", type=str, default="small", help="small, large, robotcar")
    parser.add_argument("--network", type=str, default="unet")
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--classes", type=int, default=400)
    parser.add_argument("--out_channels", type=int, default=512)
    parser.add_argument("--root", type=str, default="/home/mifs/fx221/data/cam_street_view")
    parser.add_argument("--train_label_path", type=str,
                        default="camvid_360_cvpr18_P2_training_data/building_only_filtered1_hand_labels_prop")
    parser.add_argument("--train_image_path", type=str, default="camvid_360_cvpr18_P2_training_data/images_hand")
    parser.add_argument("--val_label_path", type=str,
                        default="camvid_360_cvpr18_P4_testing_data/building_only_filtered1_hand_labels_prop")
    parser.add_argument("--val_image_path", type=str, default="camvid_360_cvpr18_P4_testing_data/images_hand")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--R", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--segloss", type=str, default='ce')
    # parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--spp", dest="spp", action="store_true", default=False)
    parser.add_argument("--classification", dest="classification", action="store_true", default=True)
    parser.add_argument("--segmentation", dest="segmentation", action="store_true", default=True)
    parser.add_argument("--hierarchical", dest="hierarchical", action="store_true", default=False)
    parser.add_argument("--val", dest="val", action="store_true", default=False)
    parser.add_argument("--aug", dest="aug", action="store_true", default=False)
    parser.add_argument("--preload", dest="preload", action="store_true", default=False)
    parser.add_argument("--ignore_bg", dest="ignore_bg", action="store_true", default=False)
    parser.add_argument("--weight_cls", type=float, default=1.0)
    parser.add_argument("--pretrained_weight", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default='timm-resnest50d')
    parser.add_argument("--encoder_weights", type=str, default='imagenet')
    parser.add_argument("--save_root", type=str, default="/home/mifs/fx221/fx221/exp/shloc/aachen")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')
    parser.add_argument("--milestones", type=list, default=[60, 80])
    parser.add_argument("--grgb_gid_file", type=str)
    parser.add_argument("--train_imglist", type=str)
    parser.add_argument("--test_imglist", type=str)
    parser.add_argument("--lr_policy", type=str, default='plateau', help='plateau, step')
    parser.add_argument("--multi_lr", type=int, default=1)
    parser.add_argument("--train_cats", type=list, default=None)
    parser.add_argument("--val_cats", type=list, default=None)

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    print('gpu: ', args.gpu)

    torch_set_gpu(gpus=args.gpu)
    main(args=args)
