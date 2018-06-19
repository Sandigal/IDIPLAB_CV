# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:14:41 2018

@author: Sandiagal
"""
import numpy as np
from scipy import misc
import random
from PIL import Image
import matplotlib.pyplot as plt


def noise(image):
    image = np.copy(image)
    height, width = image.shape[:2]
    for i in range(int(0.005*height*width)):
        x = np.random.randint(0, height)
        y = np.random.randint(0, width)
        image[x, y, :] = 255
    return image


def random_crop(image):
    height, width = image.shape[:2]

    xx = 140
    yy = 140

    random_array = np.random.random((1))
    x = int(random_array*(width-xx))
    random_array = np.random.random((1))
    y = int(random_array*(height-yy))

    image_crop = image[y:y+yy, x:x+xx, :]
#    image_crop = misc.imresize(image_crop, image.shape)

    return image_crop


def random_crop2(image, crop_shape, padding=None):
    oshape = np.shape(image)

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)

        npad = ((padding, padding), (padding, padding), (0, 0))

        image_pad = np.lib.pad(image, pad_width=npad,
                               mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        image_crop = image_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]

        return image_crop
    else:
        print("WARNING!!! nothing to do!!!")
        return image

def noise(image):
    image = np.copy(image)
    height, width = image.shape[:2]
    for i in range(int(0.005*height*width)):
        x = np.random.randint(0, height)
        y = np.random.randint(0, width)
        image[x, y, :] = 255
    return image



