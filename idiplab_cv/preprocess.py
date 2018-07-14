# -*- coding: utf-8 -*-
"""
该模块:`dataset_io`包含读取数据集以及数据集分割的类和函数。
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0

import numpy as np
from skimage.util import random_noise


class RandomNoise(object):

    def __init__(self,
                 mode='gaussian',
                 seed=None,
                 clip=True,
                 **kwargs):
        """
        Args:
            param1 (int): The first parameter.
            param2 (:obj:`str`, optional): The second parameter. Defaults to None.
                Second line of description should be indented.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def __call__(self, img):
        img = img/255
        img = random_noise(img,
                           mode=self.mode,
                           seed=self.seed,
                           clip=self.clip,
                           **self.kwargs)
        return img


class Imgaug(object):

    def __init__(self, Sequential):
        self.Sequential = Sequential

    def __call__(self, img):
        img = np.expand_dims(img, axis=0)
        images_aug = self.Sequential.augment_images(img)
        return images_aug


class BaseCrop(object):

    def __init__(self,
                 mode='center_crop',
                 crop_size=(100, 100),
                 seed=None,
                 **kwargs):
        self.mode = mode
        self.crop_size = crop_size
        self.seed = seed
        self.kwargs = kwargs

    def center_crop(self, img, crop_size, **kwargs):
        centerw, centerh = img.shape[0]//2, img.shape[1]//2
        halfw, halfh = crop_size[0]//2, crop_size[1]//2
        return img[centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh, :]

    def random_crop(self, img, crop_size, seed=None, **kwargs):
        np.random.seed(seed)
        w, h = img.shape[0], img.shape[1]
        rangew = (w - crop_size[0]) // 2 if w > crop_size[0] else 0
        rangeh = (h - crop_size[1]) // 2 if h > crop_size[1] else 0
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        return img[offsetw:offsetw+crop_size[0], offseth:offseth+crop_size[1], :]

    def __call__(self, img):
        shape = img.shape
        if self.mode is 'center_crop':
            img = self.center_crop(img, self.crop_size, **self.kwargs)
        elif self.mode is 'random_crop':
            img = self.random_crop(img, self.crop_size,
                                   self.seed, **self.kwargs)
#        img = misc.imresize(img, shape)
        return img
