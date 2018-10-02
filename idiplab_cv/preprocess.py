# -*- coding: utf-8 -*-
"""
该模块:`dataset_io`包含读取数据集以及数据集分割的类和函数。

Note:
        请注意 :meth:`models` 中的 `input_shape` 的格式为 `(height, width, channel)`。
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
        """随机噪声。

        调用 ``skimage.util.random_noise`` 来产生随机噪声。

        Args:
            mode (:obj:`str`): 只能接受以下代表噪声方法的字符串之一。
            ‘gaussian’ Gaussian-distributed additive noise.
            ‘localvar’ Gaussian-distributed additive noise, with specified local variance at each point of image
            ‘poisson’ Poisson-distributed noise generated from the data.
            ‘salt’ Replaces random pixels with 1.
            ‘pepper’ Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).
            ‘s&p’ Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
            ‘speckle’ Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
            seed (:obj:`int`, 可选): 随机数种子。默认为 ``None``。
            clip (:obj:`bool`, 可选): 加入噪声后，像素值若超过图像数据范围，将其限制在范围内。默认为 ``True``。
            **kwargs (:obj:`args`, 可选): 其他参数，可以参照 skimage.util.random_noise_。

            .. _skimage.util.random_noise: http://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise

        """
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def __call__(self, img):
        print(type(img))
        print(img.shape)
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
