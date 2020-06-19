# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagal

测试idiplab_cv.augment的各种API

"""

import idiplab_cv.augment as agmt
from idiplab_cv import preprocess

from imgaug import augmenters as ia

# %% 基于图像变换的数据增强

agmtgen = agmt.AugmentGenerator(path="data/tmp")

seq = ia.Sometimes(
    1,
    ia.Sequential([
#        ia.Fliplr(0.5),
#
#        ia.CropAndPad(
#            percent=(0, 0.1),
#            pad_mode=["constant", "edge"],
#            pad_cval=(0)
#        ),

#        ia.Sometimes(
#            1,
#            ia.OneOf([
#                ia.GaussianBlur((0, 1.0)),
#                ia.AverageBlur(k=(1, 4)),
#                ia.Sharpen(alpha=(0.0, 0.5), lightness=(0.8, 1.2))
#            ])
#        ),

        ia.AdditiveGaussianNoise(scale=(0.0, 0.75*255), per_channel=True),

#        ia.Affine(
#            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
#            rotate=(-10, 10),
#            shear=(-5, 5),
#            mode=["constant", "edge"],
#            cval=(0)
#        )

    ], random_order=True))
preprocessor = preprocess.Imgaug(seq)

datagen_args = dict(
#    brightness_range=[0.8, 1.2],
#    channel_shift_range=10,
    preprocessing_function=preprocessor,
)
agmtgen.normol_augment(datagen_args=datagen_args, augment_amount=20)


# %% 基于监督裁剪的数据增强

import numpy as np

# 原始图像
import idiplab_cv.dataset_io as io
dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
        path="nuclear_deblur_336x224.h5")
imgs_origin = np.array(dataset.imgs_origin)
labels_origin = dataset.labels_origin
names_origin = dataset.names_origin

# 图像预处理
from pickle import load
f = open("Record/20190107_0_result.h5", "rb")
contact = load(f)
mean = contact["mean"]
std = contact["std"]
f.close()
imgs_white = (imgs_origin-mean)/std

# 读取模型
from keras.models import load_model
model = load_model('Record/20190107_0_model.h5')

# 初始化增强器
cropgen = agmt.cropGenerator(
    imgs_origin[:5], imgs_white[:5], labels_origin[:5], names_origin[:5])

# 监督裁剪
feature_layer = 'activation_98'
weight_layer = 'predictions'
cropgen.AB_crop("data/nuclear_去模糊-349_480x320", model, feature_layer=feature_layer,
             weight_layer=weight_layer, SC_crop=True, augment_amount=10)
