# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> test
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   2021-05-31 15:09
=================================================='''
import argparse
import json
import torchvision.transforms as tvf
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import os
import numpy as np
import os.path as osp
from net.segnet import get_segnet
from tools.common import torch_set_gpu
from tools.seg_tools import read_seg_map_without_group, label_to_bgr

val_transform = tvf.Compose(
    (
        # tvf.ToPILImage(),
        # tvf.Resize(224),
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    )
)


def predict(net, img):
    img_tensor = val_transform(img)
    img_tensor = img_tensor.cuda().unsqueeze(0)
    with torch.no_grad():
        prediction = net(img_tensor)

    return prediction


def inference_rec(output, img, fn, map_gid_rgb, save_dir=None):
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    with torch.no_grad():
        # output = predict(net=model, img=img)
        pred_mask = output["masks"][0]
        pred_label = torch.softmax(pred_mask, dim=1).max(1)[1].cpu().numpy()
        pred_conf = torch.softmax(pred_mask, dim=1).max(1)[0].cpu().numpy()

        last_feat = output['feats'][-1]
        pred_feat_max = F.adaptive_max_pool2d(last_feat, output_size=(1, 1))
        pred_feat_avg = F.adaptive_avg_pool2d(last_feat, output_size=(1, 1))

    if pred_label.shape[0] == 1:
        pred_label = pred_label[0]
        pred_conf = pred_conf[0]

    uids = np.unique(pred_label).tolist()
    pred_seg = label_to_bgr(label=pred_label, maps=map_gid_rgb)
    pred_conf_img = np.uint8(pred_conf * 255)
    pred_conf_img = cv2.applyColorMap(src=pred_conf_img, colormap=cv2.COLORMAP_PARULA)

    H = args.R  # img.shape[0]
    W = args.R  # img.shape[1]
    pred_seg = cv2.resize(pred_seg, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
    pred_conf_img = cv2.resize(pred_conf_img, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, dsize=(W, H))

    img_seg = (0.5 * img + 0.5 * pred_seg).astype(np.uint8)
    cat_img = np.hstack([img_seg, pred_seg, pred_conf_img])

    cv2.imshow("out", cat_img)
    key = cv2.waitKey()
    if key in (27, ord('q')):  # exit by pressing key esc or q
        cv2.destroyAllWindows()
        exit(0)
    # return

    if save_dir is not None:
        conf_fn = osp.join(save_dir, "confidence", fn.split('.')[0] + ".npy")
        mask_fn = osp.join(save_dir, "masks", fn.replace("jpg", "png"))
        vis_fn = osp.join(save_dir, "vis", fn.replace("jpg", "png"))
        if not osp.exists(osp.dirname(conf_fn)):
            os.makedirs(osp.dirname(conf_fn), exist_ok=True)
        if not osp.exists(osp.dirname(vis_fn)):
            os.makedirs(osp.dirname(vis_fn), exist_ok=True)
        if not osp.exists(osp.dirname(mask_fn)):
            os.makedirs(osp.dirname(mask_fn), exist_ok=True)

        pred_confidence, pred_ids = torch.topk(torch.softmax(pred_mask, dim=1), k=10, largest=True, dim=1)
        conf_data = {"confidence": pred_confidence[0].cpu().numpy(),
                     "ids": pred_ids[0].cpu().numpy(),
                     'feat_max': pred_feat_max.squeeze().cpu().numpy(),
                     'feat_avg': pred_feat_avg.squeeze().cpu().numpy(),
                     }

        np.save(conf_fn, conf_data)
        cv2.imwrite(vis_fn, cat_img)
        cv2.imwrite(mask_fn, pred_seg)


def main(args):
    map_gid_rgb = read_seg_map_without_group(args.grgb_gid_file)

    model = get_segnet(network=args.network,
                       n_classes=args.classes,
                       encoder_name=args.encoder_name,
                       encoder_weights=args.encoder_weights,
                       encoder_depth=args.encoder_depth,
                       upsampling=args.upsampling,
                       out_channels=args.out_channels,
                       classification=args.classification,
                       segmentation=args.segmentation, )
    print("model: ", model)
    if args.pretrained_weight is not None:
        model.load_state_dict(torch.load(args.pretrained_weight), strict=True)
        print("Load weight from {:s}".format(args.pretrained_weight))
    model.eval().cuda()

    img_path = args.image_path
    save_dir = args.save_dir

    print('Save results to ', save_dir)

    imglist = []
    with open(args.image_list, "r") as f:
        lines = f.readlines()
        for l in lines:
            imglist.append(l.strip())

    for fn in tqdm(imglist, total=len(imglist)):
        if fn.find('left') >= 0 or fn.find('right') >= 0:
            continue
        img = cv2.imread(osp.join(img_path, fn))
        img = cv2.resize(img, dsize=(args.R, args.R))

        with torch.no_grad():
            output = predict(net=model, img=img)
            inference_rec(output=output, img=img, fn=fn, map_gid_rgb=map_gid_rgb, save_dir=save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test Semantic localization Network")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--pretrained_weight", type=str, default=None)
    parser.add_argument("--save_root", type=str, default="/home/mifs/fx221/fx221/exp/shloc/aachen")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--network", type=str, default="pspf")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default='timm-resnest50d')
    parser.add_argument("--encoder_weights", type=str, default='imagenet')
    parser.add_argument("--out_channels", type=int, default='2048')
    parser.add_argument("--upsampling", type=int, default='8')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')
    parser.add_argument("--R", type=int, default=256)

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    torch_set_gpu(gpus=args.gpu)
    main(args=args)
