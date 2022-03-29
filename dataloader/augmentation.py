# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> augmentation
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   2021-06-03 10:32
=================================================='''

import random
import numbers
import math
import collections
import cv2
import os.path as osp
import torchvision.transforms as tvf

from PIL import ImageOps, Image
import numpy as np


class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img):
        return ImageOps.expand(img, border=self.pad, fill=0)


class Scale:
    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img, target = imgmap
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img, target
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation), target.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation), target.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation), target.resize(self.size, self.interpolation)


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img, target = imgmap
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), target.crop((x1, y1, x1 + tw, y1 + th))


class RandomScale:
    def __init__(self, low=0.5, high=2., interpolation=Image.NEAREST):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img, target = imgmap
        scale = random.uniform(self.low, self.high)
        nw = int(img.size[0] * scale)
        nh = int(img.size[1] * scale)
        return img.resize((nw, nh), self.interpolation), \
               target.resize((nw, nh), self.interpolation)


class RandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img, target = imgmap
        w, h = img.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return img, target
            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
            return img.crop((x1, y1, x1 + tw, y1 + th)), target.crop((x1, y1, x1 + tw, y1 + th))
        else:
            return img, target


class RandomSizedCrop:

    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img, target = imgmap
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.5, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                target = target.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))
                assert (target.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation), \
                       target.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale((img, target)))


class RandomHorizontalFlip:

    def __call__(self, imgmap):
        img, target = imgmap
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
        return img, target


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, imgmap):
        img, target = imgmap
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (self.radius, self.radius), 0)
        return img, target


class RandomRotation:
    def __init__(self, degree=10):
        self.degree = degree

    def __call__(self, imgmap):
        img, target = imgmap
        deg = np.random.randint(-self.degree, self.degree, 1)[0]
        return img.rotate(deg), target.rotate(deg)


class ToPILImage:
    def __call__(self, imgmap):
        img, target = imgmap
        return tvf.ToPILImage()(img), tvf.ToPILImage()(target)


class Resize:
    def __init__(self, size=256, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        img, target = imgmap
        return img.resize((self.size, self.size), self.interpolation), \
               target.resize((self.size, self.size), self.interpolation)


class ToNumpy:
    def __call__(self, imgmap):
        img, target = imgmap
        return np.array(img), np.array(target)


class Augmentation:
    def __init__(self,
                 fovs=[45, 60, 75, 90, 105, 120, 135, 150]):
        self.fovs = fovs

    def __call__(self, img, label, min_theta, max_theta, min_phi, max_phi, res_x=256, res_y=256):
        if max_theta - min_theta < 1:
            theta = min_theta
        else:
            theta = np.random.randint(0, np.ceil(max_theta - min_theta)) + min_theta
        if max_phi - min_phi < 1:
            phi = min_phi
        else:
            phi = np.random.randint(0, np.ceil(max_phi - min_phi)) + min_phi

        # print('theta: ', min_theta, max_theta, theta)
        # print('phi: ', min_phi, max_phi, phi)
        theta = theta / 180. * math.pi
        phi = phi / 180. * math.pi

        id = np.random.randint(0, len(self.fovs))
        # fov = self.fovs[id]
        fov = 90

        # print(theta, phi, fov)

        map_y, map_x = self.crop_panorama_image(img_x=img.shape[0],
                                                img_y=img.shape[1],
                                                theta=theta, phi=phi, fov=fov, res_x=256, res_y=256)

        crop_img = cv2.remap(img, map_y, map_x, cv2.INTER_AREA)
        crop_label = cv2.remap(label, map_y, map_x, cv2.INTER_NEAREST)

        return crop_img, crop_label

    def crop_panorama_image(slef, img_x, img_y, theta=0.0, phi=0.0, res_x=512, res_y=512, fov=60.0):
        # img_x = img.shape[0]
        # img_y = img.shape[1]

        theta = theta / 180 * math.pi
        phi = phi / 180 * math.pi

        fov_x = fov
        aspect_ratio = res_y * 1.0 / res_x
        half_len_x = math.tan(fov_x / 180 * math.pi / 2)
        half_len_y = aspect_ratio * half_len_x

        pixel_len_x = 2 * half_len_x / res_x
        pixel_len_y = 2 * half_len_y / res_y

        axis_y = math.cos(theta)
        axis_z = math.sin(theta)
        axis_x = 0

        # theta rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        theta_rot_mat = np.array([[1, 0, 0], \
                                  [0, cos_theta, -sin_theta], \
                                  [0, sin_theta, cos_theta]], dtype=np.float32)

        # phi rotation matrix
        cos_phi = math.cos(phi)
        sin_phi = -math.sin(phi)
        phi_rot_mat = np.array([[cos_phi + axis_x ** 2 * (1 - cos_phi), \
                                 axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
                                 axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
                                [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
                                 cos_phi + axis_y ** 2 * (1 - cos_phi), \
                                 axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
                                [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
                                 axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
                                 cos_phi + axis_z ** 2 * (1 - cos_phi)]], dtype=np.float32)

        map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
        map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

        map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
        map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
        map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

        ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
                                         np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

        ind = theta_rot_mat.dot(ind)
        ind = phi_rot_mat.dot(ind)

        vec_len = np.sqrt(np.sum(ind ** 2, axis=0))
        ind /= np.tile(vec_len, (3, 1))

        cur_phi = np.arcsin(ind[0, :])
        cur_theta = np.arctan2(ind[1, :], -ind[2, :])

        map_x = (cur_phi + math.pi / 2) / math.pi * img_x
        map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

        map_x = np.reshape(map_x, [res_x, res_y])
        map_y = np.reshape(map_y, [res_x, res_y])

        return map_y, map_x
        # return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


class Perspective:
    def __init__(self,
                 # fovs=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150],
                 fovs=[30, 60, 90, 120, 150],
                 thetas=[0, 30, 60, 90, 120, 150, 180,
                         -30, -60, -90, -120, -150, -180],
                 # phis=[0, 5, 10, 15, 20, 25, 30, -5, -10, -15, -20, -25, -30]
                 phis=[0, 5, 10, 15, -5, -10, -15, ]
                 ):
        self.fovs = fovs
        self.thetas = thetas
        self.phis = phis

    def __call__(self, img, label, res_x=256, res_y=256):
        id = np.random.randint(0, len(self.thetas))
        theta = self.thetas[id]
        # theta = (cur_theta + math.pi) % (2 * math.pi) - math.pi

        id = np.random.randint(0, len(self.phis))
        phi = self.phis[id]

        id = np.random.randint(0, len(self.fovs))
        fov = self.fovs[id]

        map_y, map_x = self.crop_panorama_image(img_x=img.shape[0], img_y=img.shape[1],
                                                theta=theta, phi=phi, fov=fov, res_x=256, res_y=256)

        crop_img = cv2.remap(img, map_y, map_x, cv2.INTER_AREA)
        crop_label = cv2.remap(label, map_y, map_x, cv2.INTER_NEAREST)

        return crop_img, crop_label

    def crop_panorama_image(slef, img_x, img_y, theta=0.0, phi=0.0, res_x=512, res_y=512, fov=60.0):
        # img_x = img.shape[0]
        # img_y = img.shape[1]

        theta = theta / 180 * math.pi
        phi = phi / 180 * math.pi

        fov_x = fov
        aspect_ratio = res_y * 1.0 / res_x
        half_len_x = math.tan(fov_x / 180 * math.pi / 2)
        half_len_y = aspect_ratio * half_len_x

        pixel_len_x = 2 * half_len_x / res_x
        pixel_len_y = 2 * half_len_y / res_y

        axis_y = math.cos(theta)
        axis_z = math.sin(theta)
        axis_x = 0

        # theta rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        theta_rot_mat = np.array([[1, 0, 0], \
                                  [0, cos_theta, -sin_theta], \
                                  [0, sin_theta, cos_theta]], dtype=np.float32)

        # phi rotation matrix
        cos_phi = math.cos(phi)
        sin_phi = -math.sin(phi)
        phi_rot_mat = np.array([[cos_phi + axis_x ** 2 * (1 - cos_phi), \
                                 axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
                                 axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
                                [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
                                 cos_phi + axis_y ** 2 * (1 - cos_phi), \
                                 axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
                                [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
                                 axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
                                 cos_phi + axis_z ** 2 * (1 - cos_phi)]], dtype=np.float32)

        map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
        map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

        map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
        map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
        map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

        ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
                                         np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

        ind = theta_rot_mat.dot(ind)
        ind = phi_rot_mat.dot(ind)

        vec_len = np.sqrt(np.sum(ind ** 2, axis=0))
        ind /= np.tile(vec_len, (3, 1))

        cur_phi = np.arcsin(ind[0, :])
        cur_theta = np.arctan2(ind[1, :], -ind[2, :])

        map_x = (cur_phi + math.pi / 2) / math.pi * img_x
        map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

        map_x = np.reshape(map_x, [res_x, res_y])
        map_y = np.reshape(map_y, [res_x, res_y])

        return map_y, map_x
        # return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def pano2img(img, move=0.5, theta=0.0, phi=0.0, res_x=256, res_y=256):
    img_x = img.shape[0]
    img_y = img.shape[1]

    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi

    mid_x = res_x / 2
    mid_y = res_y / 2

    axis_y = math.cos(theta)
    axis_z = math.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    theta_rot_mat = np.array([[1, 0, 0], \
                              [0, cos_theta, -sin_theta], \
                              [0, sin_theta, cos_theta]], dtype=np.float32)

    # phi rotation matrix
    cos_phi = math.cos(phi)
    sin_phi = -math.sin(phi)
    phi_rot_mat = np.array([[cos_phi + axis_x ** 2 * (1 - cos_phi), \
                             axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
                             axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
                            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
                             cos_phi + axis_y ** 2 * (1 - cos_phi), \
                             axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
                            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
                             axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
                             cos_phi + axis_z ** 2 * (1 - cos_phi)]], dtype=np.float32)

    indx = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    indy = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = np.sin(indx * math.pi / res_x - math.pi / 2)
    map_y = np.sin(indy * (2 * math.pi) / res_y) * np.cos(indx * math.pi / res_x - math.pi / 2)
    map_z = -np.cos(indy * (2 * math.pi) / res_y) * np.cos(indx * math.pi / res_x - math.pi / 2)

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
                                     np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    move_dir = np.array([0, 0, -1], dtype=np.float32)
    move_dir = theta_rot_mat.dot(move_dir)
    move_dir = phi_rot_mat.dot(move_dir)

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    ind += np.tile(move * move_dir, (ind.shape[1], 1)).T

    vec_len = np.sqrt(np.sum(ind ** 2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + math.pi / 2) / math.pi * img_x
    map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    out = cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR)
    return out


def crop_panorama_image(img_x, img_y, theta=0.0, phi=0.0, res_x=512, res_y=512, fov=60.0, debug=False):
    # img_x = img.shape[0]
    # img_y = img.shape[1]

    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi

    fov_x = fov
    aspect_ratio = res_y * 1.0 / res_x
    half_len_x = math.tan(fov_x / 180 * math.pi / 2)
    half_len_y = aspect_ratio * half_len_x

    pixel_len_x = 2 * half_len_x / res_x
    pixel_len_y = 2 * half_len_y / res_y

    axis_y = math.cos(theta)
    axis_z = math.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    theta_rot_mat = np.array([[1, 0, 0], \
                              [0, cos_theta, -sin_theta], \
                              [0, sin_theta, cos_theta]], dtype=np.float32)

    # phi rotation matrix
    cos_phi = math.cos(phi)
    sin_phi = -math.sin(phi)
    phi_rot_mat = np.array([[cos_phi + axis_x ** 2 * (1 - cos_phi), \
                             axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
                             axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
                            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
                             cos_phi + axis_y ** 2 * (1 - cos_phi), \
                             axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
                            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
                             axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
                             cos_phi + axis_z ** 2 * (1 - cos_phi)]], dtype=np.float32)

    map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
    map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
    map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
                                     np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    vec_len = np.sqrt(np.sum(ind ** 2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + math.pi / 2) / math.pi * img_x
    map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    if debug:
        for x in range(res_x):
            for y in range(res_y):
                print('(%.2f, %.2f)\t' % (map_x[x, y], map_y[x, y]))
            print()

    return map_y, map_x
    # return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def key_press(key, mod):
    intv = 6
    global cur_phi, cur_theta, stop_requested, img_updated
    if key == ord('Q') or key == ord('q'):  # q/Q
        stop_requested = True
    if key == 0xFF52:  # up
        if cur_phi > -80.0 / 180 * math.pi:
            cur_phi -= math.pi / intv
        print('Theta: %.4f, Phi: %.4f' % (cur_theta, cur_phi))
        img_updated = True
    if key == 0xFF53:  # right
        cur_theta += math.pi / intv
        cur_theta = (cur_theta + math.pi) % (2 * math.pi) - math.pi
        print('Theta: %.4f, Phi: %.4f' % (cur_theta, cur_phi))
        img_updated = True
    if key == 0xFF51:  # left
        cur_theta -= math.pi / intv
        cur_theta = (cur_theta + math.pi) % (2 * math.pi) - math.pi
        print('Theta: %.4f, Phi: %.4f' % (cur_theta, cur_phi))
        img_updated = True
    if key == 0xFF54:  # down
        if cur_phi < 80.0 / 180 * math.pi:
            cur_phi += math.pi / intv
        print('Theta: %.4f, Phi: %.4f' % (cur_theta, cur_phi))
        img_updated = True


if __name__ == '__main__':

    trans = [
        ToPILImage(),
        RandomRotation(degree=10),
        RandomSizedCrop(size=256),
        RandomHorizontalFlip(),
        ToNumpy(), ]

    img_seg_gid = cv2.imread("../1417176584890937.png")
    img = img_seg_gid[:, :480]
    seg = img_seg_gid[:, 480: 480 * 2]
    gid = img_seg_gid[:, 480 * 2: 480 * 3]
    cv2.imshow("img", img_seg_gid)
    # trans_img =

    for t in trans:
        img, gid = t((img, gid))
        # img.show()
        # seg.show()
        print(type(img), type(gid))
        show_img = np.array(img).astype(np.uint8)
        show_gid = np.array(gid).astype(np.uint8)
        cv2.imshow("trans_img", show_img)
        cv2.imshow("trans_gid", show_gid)
        cv2.waitKey(0)

    exit(0)
    save_dir = '/data/cornucopia/fx221/exp/shloc/crop'
    root_dir = '/home/mifs/fx221/data/cam_street_view/cambridge_center_medium'
    # fn = 'road_00008209_00000028.png'
    # fn = 'road_00008158_00000219.png'
    # fn = 'road_00005295_00000112.png'
    # fn = 'road_00000004_00000006.png'
    fn = 'road_00000096_00000026.png'
    pano_img = cv2.imread(osp.join(root_dir, 'images', fn))
    pano_img = cv2.resize(pano_img, dsize=(1920, 960))
    label_img = cv2.imread(osp.join(root_dir, 'segm_from_cubes', fn))
    # print(label_img.shape, pano_img.shape)
    # exit(0)

    cv2.imshow('pano', pano_img)
    cur_theta = 0.0
    cur_phi = 0.0
    curr_fov = 90
    intv = 12

    stop_requested = False
    img_updated = True

    Pesp = Perspective()
    # for ta in Pesp.thetas:
    # for ta in Pesp.phis:
    for ta in Pesp.fovs:
        # cur_theta = ta
        # cur_phi = ta
        curr_fov = ta
        map_y, map_x = crop_panorama_image(pano_img.shape[0], pano_img.shape[1], cur_theta, cur_phi,
                                           fov=curr_fov, res_x=256, res_y=256)
        crop_img = cv2.remap(pano_img, map_y, map_x, cv2.INTER_AREA)
        crop_label = cv2.remap(label_img, map_y, map_x, cv2.INTER_NEAREST)
        crop = np.vstack([crop_img, crop_label])
        # cv2.imshow('crop_img', crop_img)
        # cv2.imshow('label', crop_label)
        cv2.imshow('crop', crop)

        cv2.imwrite(osp.join(save_dir,
                             '{:s}_{:d}_{:.2f}_{:.2f}.png'.format(fn.split('.')[0], curr_fov, cur_theta, cur_phi)),
                    crop)
        cv2.waitKey(0)

    exit(0)

    while True:
        if stop_requested:
            break

        if img_updated:
            img_updated = False
            print('fov: ', curr_fov, 'Theta: ', cur_theta, 'Phi: ', cur_phi)
            map_y, map_x = crop_panorama_image(pano_img, cur_theta / math.pi * 180, cur_phi / math.pi * 180,
                                               fov=curr_fov, res_x=256, res_y=256)
            crop_img = cv2.remap(pano_img, map_y, map_x, cv2.INTER_AREA)
            crop_label = cv2.remap(label_img, map_y, map_x, cv2.INTER_NEAREST)
            crop = np.vstack([crop_img, crop_label])
            # cv2.imshow('crop_img', crop_img)
            # cv2.imshow('label', crop_label)
            cv2.imshow('crop', crop)

            cv2.imwrite(osp.join(save_dir,
                                 '{:s}_{:d}_{:.2f}_{:.2f}.png'.format(fn.split('.')[0], curr_fov, cur_theta, cur_phi)),
                        crop)

        while True:
            key = cv2.waitKey()
            print(key)
            if key == 119:  # w
                curr_fov += 15
                img_updated = True
                break
            elif key == 115:  # s
                curr_fov -= 15
                img_updated = True
                break
            elif key == 27:
                exit(0)
            elif key == 117:  # up u
                if cur_phi > -80.0 / 180 * math.pi:
                    cur_phi -= math.pi / intv
                print('Theta: %.4f, Phi: %.4f' % (cur_theta, cur_phi))
                img_updated = True
                break
            elif key == 114:  # right, r
                cur_theta += math.pi / intv
                cur_theta = (cur_theta + math.pi) % (2 * math.pi) - math.pi
                print('Theta: %.4f, Phi: %.4f' % (cur_theta, cur_phi))
                img_updated = True
                break
            elif key == 108:  # left, l
                cur_theta -= math.pi / intv
                cur_theta = (cur_theta + math.pi) % (2 * math.pi) - math.pi
                print('Theta: %.4f, Phi: %.4f' % (cur_theta, cur_phi))
                img_updated = True
                break
            elif key == 100:  # down, d
                if cur_phi < 80.0 / 180 * math.pi:
                    cur_phi += math.pi / intv
                print('Theta: %.4f, Phi: %.4f' % (cur_theta, cur_phi))
                img_updated = True
                break

# if __name__ == '__main__':
