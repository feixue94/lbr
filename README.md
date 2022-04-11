# Efficient Large-scale Localization by Global Instance Recognition

<p align="center">
  <img src="assets/overview.png" width="1024">
</p>

* Full paper PDF: [Efficient Large-scale Localization by Global Instance Recognition](https://arxiv.org/abs/1911.11763).

* Authors: *Fei Xue, Ignas Budvytis, Daniel Olmeda Reino, Roberto Cipolla*

* Website: [lbr](https://github.com/feixue94/feixue94.github.io/lbr) for videos, slides, recent updates, and datasets.

## Dependencies

* Python 3 >= 3.6
* PyTorch >= 1.8
* OpenCV >= 3.4
* NumPy >= 1.18
* segmentation-models-pytorch = 0.1.3
* colmap
* pycolmap = 0.0.1

## Data preparation

Please follow instructions on the [VisualLocalization Benchmark](https://www.visuallocalization.net/datasets/) to
download images Aachen and RobotCar-Seasons datasets

* [Images of Aachen_v1.1 dataset](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/)
* [Images of RobotCar-Seasons dataset](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/)
* Global instances of Aachen_v1.1 dataset
* Global instances of RobotCar-Seasons dataset

Since only daytime images are included in Aachen and RobotCar-Seasons database, which may cause recognition loss for
recognition of nighttime query images, we augment the training data by generating some stylized images. The structure of
files in Aachen_v1.1 dataset should be like this:

```
- aachen_v1.1
    - global_seg_instance
       - db
       - query
       - sequences
    - images
    - 3D-models
      - images.bin
      - points3D.bin
      - cameras.bin
    - stylized
      - images (raw database images)
      - images_aug1
      - images_aug2
```

For RobotCar-Seasons dataset, it should be like this:

```
- RobotCar-Seasons
  - gloabl_seg_instance
    - overcast-reference
        - rear
  - images
    - overcast-reference
    - night
    ...
    - night-rain
  - 3D-models 
    - sfm-sift
        - cameras.bin
        - images.bin
        - points3D.bin
  - stylized
    - overcast-reference (raw database images)
    - overcast-reference_aug_d 
    - overcast-reference_aug_r 
    - overcast-reference_aug_nr 
```

## Pretrained weights

We provide the pretrained weights for local feature detection and extraction, global instance recognition for
Aachen_v1.1 and RobotCar-Seasons datasets.

* [Local feature](https://drive.google.com/file/d/1N4j7PkZoy2CkWhS7u6dFzMIoai3ShG9p/view?usp=sharing)
* [Recognition for Aachen_v1.1](https://drive.google.com/file/d/17qerRcU8Iemjwz7tUtlX9syfN-WVSs4K/view?usp=sharing)
* [Recognition for RobotCar-Seasons](https://drive.google.com/file/d/1Ns5I3YGoMCBURzWKZTxqsugeG4jUcj4a/view?usp=sharing)

## 3D Reconstruction

* feature extraction and 3d reconstruction for Aachen_v1.1

```
./run_reconstruct_aachen
```

* feature extraction and 3d reconstruction for RobotCar-Seasons

```
./run_reconstruct_robotcar
```

## Localization with global instances

* localization on Aachen_v1.1

```
./run_loc_aachn
```

you will get results like this:

|          | Day  | Night       | 
| -------- | ------- | -------- |
| cvpr | 89.1 / 96.1 / 99.3 | 77.0 / 90.1 / 99.5  |
| post-cvpr | 88.8 / 95.8 / 99.2 | 75.4 / 91.6 / 100 |

* localization on RobotCar-Seasons

```
./run_loc_robotcar
```

you will get results like this:

|        | Night  | Night-rain       | 
| -------- | ----- | ------- |
| cvpr | 24.9 / 62.3 / 86.1 | 47.5 / 73.4 / 90.0  |
| post-cvpr | 28.1 / 66.9 / 91.8 | 46.1 / 73.6 / 92.5 |

## Testing of global instance recognition
you will get predicted masks of global instances, confidence maps, global features, and visualization images.

* testing recognition on Aachen_v1.1

```
./test_aachen
```

<p align="center">
  <img src="assets/samples/1116.png" width="1024">
</p>

* testing recognition on RobotCar-Seasons

```
./test_robotcar
```

<p align="center">
  <img src="assets/samples/1417176916116788.png" width="1024">
</p>


## Training

* training recognition on Aachen_v1.1

```
./train_aachen
```

* training recognition on RobotCar-Seasons

```
./train_robotcar
```

## BibTeX Citation

If you use any ideas from the paper or code from this repo, please consider citing:

```
@inproceedings{xue2022efficient,
  author    = {Fei Xue and Ignas Budvytis and Daniel Olmeda Reino and Roberto Cipolla},
  title     = {Efficient Large-scale Localization by Global Instance Recognition},
  booktitle = {CVPR},
  year      = {2022}
}
```

## Acknowledgements

Part of the code is from previous excellent works including [SuperPoint](), [R2D2](https://github.com/naver/r2d2)
, [hloc](https://github.com/cvg/Hierarchical-Localization). If you can find more details from their released
repositories if you are interested in their works. 
