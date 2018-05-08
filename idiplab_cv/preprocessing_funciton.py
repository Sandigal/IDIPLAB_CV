# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:14:41 2018

@author: Sandiagal
"""
import numpy as np
from scipy import misc

def noise(image):
    height, width = image.shape[:2]

    for i in range(int(0.0005*height*width)):
        x = np.random.randint(0, height)
        y = np.random.randint(0, width)
        image[x, y, :] = 255

    return image

def random_crop_with_noise(image):
    height, width = image.shape[:2]

    random_array = np.random.random((1))
    x = int((1+0.2*(random_array-0.5))*(width+height)/2)
    w = height
    image_crop = image[:, x:w+x, :]
    image_crop = misc.imresize(image_crop, image.shape)

    for i in range(int(0.0001*height*width)):
        x = np.random.randint(0, height)
        y = np.random.randint(0, width)
        image[x, y, :] = 255

    return image
