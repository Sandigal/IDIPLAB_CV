# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagal

分别读取每个类下的单个文件，用flow对单个图片分别处理
输出图像分类到对应类文件夹内
保证每张图片数量相同
保证增强后图像的前缀和原图一致
失去多线程的优势

"""

import augment as agmt
import preprocess

from imgaug import augmenters as ia

# %%

agmtgen = agmt.AugmentGenerator(path="../data/tmp")

seq = ia.Sequential([

        ia.Fliplr(0.5),

        ia.CropAndPad(
            percent=(0, 0.2),
            pad_mode=["constant", "edge"],
            pad_cval=(0)
        ),

        ia.Sometimes(
            1,
            ia.OneOf([
                ia.GaussianBlur((0, 5.0)),
                ia.AverageBlur(k=(2, 11)),
                ia.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5))
            ])
        ),

        ia.OneOf([
            ia.AdditiveGaussianNoise(scale=(0.0, 0.1*255)),
            ia.CoarseDropout(0.1, size_percent=0.2)
        ]),

        ia.OneOf([
            ia.Add((-50, 0)),
            ia.Multiply((1, 1.5))
        ]),

        ia.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-20, 20),
            shear=(-10, 10),
            mode=["constant", "edge"],
            cval=(0)
        )

], random_order=True)  # apply augmenters in random order

#preprocessor = preprocess.RandomNoise(mode="salt",amount=0.005)
#
# datagen_args = dict(
#    rotation_range=15.,
#    width_shift_range=0.05,
#    height_shift_range=0.05,
#    shear_range=10.,
#    zoom_range=0.1,
#    channel_shift_range=5.,
#    horizontal_flip=True,
#    preprocessing_function=preprocessor)

preprocessor = preprocess.Imgaug(seq)

datagen_args = dict(
    preprocessing_function=preprocessor)

agmtgen.normol_augment(datagen_args=datagen_args, augment_amount=10)


# %%

import numpy as np
import dataset_io as io
from keras.models import load_model
from keras.applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope
from pickle import load


path = "../data/dataset 480x320"
shape = (384, 256)
dataset = io.Dataset(path, shape=shape)
class_to_index, sample_per_class = dataset.load_data()

imgs_origin = np.array(dataset.imgs_origin)
labels_origin = dataset.labels_origin
names_origin = dataset.names_origin
del shape

mean, std = load(open('mean-std.json', 'rb'))
imgs_white = (imgs_origin-mean)/std
del mean, std

with CustomObjectScope({'relu6': relu6}):
    model = load_model('2018-06-22 model.h5')
model.summary()

cropgen = agmt.cropGenerator(
    imgs_origin, imgs_white, labels_origin, names_origin)

active_layer = 'conv_pw_13_relu'
weight_layer = 'conv_preds'
cropgen.crop(path, model, active_layer=active_layer,
             weight_layer=weight_layer, supervised_crop=True, augment_amount=10)

# %%

path = '../data/dataset 480x320'
augment_amount = 1  # 一张图成倍增长数量
shape = (224, 224)
supervised_crop = True

datagen_args = dict(
    rotation_range=3.,
    width_shift_range=0.03,
    height_shift_range=0.03,
    shear_range=3.,
    zoom_range=0.03,
    channel_shift_range=3.,
    horizontal_flip=True,
    preprocessing_function=preprocess.peppersalt)

agmtgen = agmt.AugmentGenerator(
    path, datagen_args, shape=shape, augment_amount=augment_amount)
agmtgen.next(supervised_crop=supervised_crop)
