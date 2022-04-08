# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> extractor
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   26/07/2021 16:49
=================================================='''
import torch
import torch.utils.data as Data

import argparse
import os
import os.path as osp
import h5py
from tqdm import tqdm
from types import SimpleNamespace
import logging
from pathlib import Path
import cv2
import numpy as np
import pprint

from localization.fine.features.extract_spp import extract_spp_return
from localization.fine.features.extract_r2d2 import extract_r2d2_return, load_network
from net.locnets.superpoint import SuperPointNet
from net.locnets.resnet import ResNetXN, extract_resnet_return

confs = {
    'resnetxn-triv2-0001-n4096-r1600-mask': {
        'output': 'feats-resnetxn-triv2-0001-n4096-r1600-mask',
        'model': {
            'name': 'resnetxn',
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': True,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(), "models/resnetxn_triv2_d128_38.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': True,
    },

    'resnetxn-triv2-0001-n3000-r1600-mask': {
        'output': 'feats-resnetxn-triv2-0001-n3000-r1600-mask',
        'model': {
            'name': 'resnetxn',
            'max_keypoints': 3000,
            'conf_th': 0.001,
            'multiscale': True,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(), "models/resnetxn_triv2_d128_38.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': True,
    },
    'resnetxn-triv2-0001-n2000-r1600-mask': {
        'output': 'feats-resnetxn-triv2-0001-n2000-r1600-mask',
        'model': {
            'name': 'resnetxn',
            'max_keypoints': 2000,
            'conf_th': 0.001,
            'multiscale': True,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(), "models/resnetxn_triv2_d128_38.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': True,
    },

    'resnetxn-triv2-0001-n1000-r1600-mask': {
        'output': 'feats-resnetxn-triv2-0001-n1000-r1600-mask',
        'model': {
            'name': 'resnetxn',
            'max_keypoints': 1000,
            'conf_th': 0.001,
            'multiscale': True,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(), "models/resnetxn_triv2_d128_38.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': True,
    },

    'resnetxn-triv2-0001-n4096-r1024-mask': {
        'output': 'feats-resnetxn-triv2-0001-n4096-r1024-mask',
        'model': {
            'name': 'resnetxn',
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': True,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(), "models/resnetxn_triv2_d128_38.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
        'mask': True,
    },

    'resnetxn-triv2-ms-0001-n4096-r1024-mask': {
        'output': 'feats-resnetxn-triv2-ms-0001-n4096-r1024-mask',
        'model': {
            'name': 'resnetxn',
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': True,
            'scales': [1.2, 1.0, 0.8],
            'model_fn': osp.join(os.getcwd(), "models/resnetxn_triv2_d128_38.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
        'mask': True,
    },

}

confs_matcher = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
        },
    },
    'NNM': {
        'output': 'NNM',
        'model': {
            'name': 'nnm',
            'do_mutual_check': True,
            'distance_threshold': None,
        },
    },
    'NNML': {
        'output': 'NNML',
        'model': {
            'name': 'nnml',
            'do_mutual_check': True,
            'distance_threshold': None,
        },
    },

    'ONN': {
        'output': 'ONN',
        'model': {
            'name': 'nn',
            'do_mutual_check': False,
            'distance_threshold': None,
        },
    },
    'NNR': {
        'output': 'NNR',
        'model': {
            'name': 'nnr',
            'do_mutual_check': True,
            'distance_threshold': 0.9,
        },
    }
}


class ImageDataset(Data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
    }

    def __init__(self, root, conf, image_list=None,
                 mask_root=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        if image_list is None:
            for g in conf.globs:
                self.paths += list(Path(root).glob('**/' + g))
            if len(self.paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            self.paths = [i.relative_to(root) for i in self.paths]
        else:
            with open(image_list, "r") as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    self.paths.append(Path(l))

        logging.info(f'Found {len(self.paths)} images in root {root}.')

        if mask_root is not None:
            self.mask_root = mask_root
        else:
            self.mask_root = None

        print("mask_root: ", self.mask_root)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(w, h) > self.conf.resize_max):
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': str(path),
            'image': image,
            'original_size': np.array(size),
        }

        if self.mask_root is not None:
            mask_path = Path(str(path).replace("jpg", "png"))
            if osp.exists(mask_path):
                mask = cv2.imread(str(self.mask_root / mask_path))
                mask = cv2.resize(mask, dsize=(image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros(shape=(image.shape[1], image.shape[2], 3), dtype=np.uint8)

            data['mask'] = mask

        return data

    def __len__(self):
        return len(self.paths)


def get_model(model_name, weight_path):
    if model_name == "superpoint":
        model = SuperPointNet().eval()
        model.load_state_dict(torch.load(weight_path))
        extractor = extract_spp_return
    elif model_name == "r2d2":
        model = load_network(model_fn=weight_path).eval()
        extractor = extract_r2d2_return
    elif model_name == 'resnetxn-ori':
        model = ResNetXN(encoder_depth=2, outdim=128).eval()
        model.load_state_dict(torch.load(weight_path)['state_dict'], strict=True)
        extractor = extract_resnet_return
    elif model_name == 'resnetxn':
        model = ResNetXN(encoder_depth=2, outdim=128).eval()
        model.load_state_dict(torch.load(weight_path)['model'], strict=True)
        extractor = extract_resnet_return

    return model, extractor


@torch.no_grad()
def main(conf, image_dir, export_dir, mask_dir=None, tag=None):
    logging.info('Extracting local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model = dynamic_load(extractors, conf['model']['name'])
    # model = Model(conf['model']).eval().to(device)
    model, extractor = get_model(model_name=conf['model']['name'], weight_path=conf["model"]["model_fn"])
    model = model.cuda()
    print("model: ", model)

    loader = ImageDataset(image_dir, conf['preprocessing'],
                          image_list=args.image_list,
                          mask_root=mask_dir)
    loader = torch.utils.data.DataLoader(loader, num_workers=4)

    feature_path = Path(export_dir, conf['output'] + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    feature_file = h5py.File(str(feature_path), 'a')

    with tqdm(total=len(loader)) as t:
        for idx, data in enumerate(loader):
            t.update()
            if tag is not None:
                if data['name'][0].find(tag) < 0:
                    continue
            pred = extractor(model, img=data["image"],
                             topK=conf["model"]["max_keypoints"],
                             mask=data["mask"][0].numpy().astype(np.uint8) if "mask" in data.keys() else None,
                             conf_th=conf["model"]["conf_th"],
                             scales=conf["model"]["scales"],
                             )

            # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            pred['descriptors'] = pred['descriptors'].transpose()

            t.set_postfix(npoints=pred['keypoints'].shape[0])
            # print(pred['keypoints'].shape)

            pred['image_size'] = original_size = data['original_size'][0].numpy()
            # pred['descriptors'] = pred['descriptors'].T
            if 'keypoints' in pred.keys():
                size = np.array(data['image'].shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

            grp = feature_file.create_group(data['name'][0])
            for k, v in pred.items():
                # print(k, v.shape)
                grp.create_dataset(k, data=v)

            del pred

    feature_file.close()
    logging.info('Finished exporting features.')

    return feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--image_list', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--mask_dir', type=Path, default=None)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superpoint_aachen',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir,
         mask_dir=args.mask_dir if confs[args.conf]["mask"] else None, tag=args.tag)
