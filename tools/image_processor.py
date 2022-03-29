# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> image_processor
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   05/10/2021 14:34
=================================================='''
import os
import os.path as osp
import numpy as np
import cv2

def unblur_filter(im):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    aug_img = cv2.filter2D(im, -1, kernel)

    return aug_img


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

if __name__ == '__main__':
    img_dir = '/scratch2/fx221/localization/RobotCar-Seasons/images/sun/rear'
    img_fns = os.listdir(img_dir)
    img_fns = sorted(img_fns)

    for fn in img_fns:
        im = cv2.imread(osp.join(img_dir, fn))
        aug_img1 = unblur_filter(im)
        aug_img2 = unsharp_mask(image=im)

        cv2.imshow('im', im)
        cv2.imshow('im_aug1', aug_img1)
        cv2.imshow('im_aug2', aug_img2)
        cv2.waitKey(-1)
