
# Efficient Large-scale Localization by Global Instance Recognition
<p align="center">
  <img src="assets/overview.png" width="500">
</p>

* Full paper PDF: [Efficient Large-scale Localization by Global Instance Recognition](https://arxiv.org/abs/1911.11763).

* Authors: *Fei Xue, Ignas Budvytis, Daniel Olmeda Reino, Roberto Cipolla*

* Website: [lbr](https://github.com/feixue94/feixue94.github.io/lbr) for videos, slides, recent updates, and datasets.

## Dependencies
* Python 3 >= 3.6
* PyTorch >= 1.8
* OpenCV >= 3.4
* NumPy >= 1.18
* segmentation-models-pytorch
* colmap
* pycolmap


## Data preparation 
Please follow instructions on the [VisualLocalization Benchmark](https://www.visuallocalization.net/datasets/) to download images  Aachen and RobotCar-Seasons datasets
* [Images of Aachen_v1.1 dataset](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/)
* [Images of RobotCar-Seasons dataset](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/)
* Global instances of Aachen_v1.1 dataset 
* Global instances of RobotCar-Seasons dataset 

## Pretrained weights
* Local feature
* Recognition for Aachen_v1.1 
* Recognition for RobotCar-Seasons 

## 3D Reconstruction 
* feature extraction and 3d reconstruction for Aachen_v1.1
```
./run_reconstruct_aachen
```

* feature extraction and 3d reconstruction for RobotCar-Seasons
```
./run_reconstruct_robotcar
```

## Testing 
* localization on Aachen_v1.1
```
./run_loc_aachn
```
* localization on RobotCar-Seasons
```
./run_loc_robotcar
```

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

```txt
@inproceedings{xue2022efficient,
  author    = {Fei Xue and Ignas Budvytis and Daniel Olmeda Reino and Roberto Cipolla},
  title     = {Efficient Large-scale Localization by Global Instance Recognition},
  booktitle = {CVPR},
  year      = {2022}
}
```
