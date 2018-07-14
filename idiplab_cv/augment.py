# -*- coding: utf-8 -*-
"""
该模块 :meth:`augment` 包含数据增强的类和函数。
详细 :ref:`数据增强`
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0


import os
from random import randint
import time

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

import dataset_io as io
import visul


class AugmentGenerator(object):
    """
        sdsdsdsdsdsd

    :param str path: 数据所在目录地址。目录结构请参照 :ref:`目录结构`。
    :param turple shape: 格式为 `(height, width, channels)`，所有的图像将被调整到该尺寸。默认:`(224, 224 3)`

    **实例化说明：**

    1. 当实例化 AugmentGenerator 时，什么都不会发生，做成类只是为了不要再一行内输入太多参数。
    """

    def __init__(self,
                 path,
                 shape=None):
        self.path = path
        self.shape = shape

    def normol_augment(self, datagen_args, augment_amount=10):
        """
        非监督数据增强。XXXXXXXXXXXXX

        :param str datagen_args: XXXXXXXX
        :param str augment_amount: 数据增广倍数，默认为10.
        """
        print("--->Start augmentation")

        datagen = ImageDataGenerator(**datagen_args)

        start = time.clock()

        is_exist = os.path.exists(
            self.path+"/augment")
        if not is_exist:
            os.makedirs(self.path+"/augment")

        # dog or cat
        sub_dir_list = os.listdir(self.path+"/origin")
        process_bar = io.ShowProcess(len(sub_dir_list))
        for sub_dir in sub_dir_list:
            process_bar.show_process(sub_dir)

            is_exist = os.path.exists(
                self.path+"/augment/"+sub_dir)
            if not is_exist:
                os.makedirs(self.path+"/augment/"+sub_dir)

            imgs, names = io.read_imgs_in_dir(
                self.path+"/origin/"+sub_dir, self.shape)
            imgs = np.array(imgs)

            # dog1 or dog2
            print("")
            process_bar = io.ShowProcess(len(names))
            for img, name in zip(imgs, names):
                process_bar.show_process()
                img = np.expand_dims(img, axis=0)

                augmentgen = datagen.flow(
                    img,
                    batch_size=1,
                    shuffle=False,
                    save_to_dir=self.path+"/augment/"+sub_dir,
                    save_format='jpg')

                for i in range(augment_amount):
                    augmentgen.save_prefix = name.split(
                        '.')[0] + "_"+str(np.random.randint(0, 99999))
                    augmentgen.next()

        end = time.clock()
        print("")
        print("Cost time:", end-start, "s")
        print("")


class cropGenerator(object):

    def __init__(self,
                 imgs_origin,
                 imgs_white,
                 labels_origin,
                 names_origin
                 ):
        self.imgs_origin = imgs_origin
        self.imgs_white = imgs_white
        self.labels_origin = labels_origin
        self.names_origin = names_origin
        self.cams = None

    def makedirs(self, path, supervised_crop=False):
        is_exist = os.path.exists(
            path+"/crop_AB")
        if not is_exist:
            os.makedirs(path+"/crop_AB")
        # dog or cat
        sub_dir_list = os.listdir(path+"/origin")
        for sub_dir in sub_dir_list:
            is_exist = os.path.exists(
                path+"/crop_AB/"+sub_dir)
            if not is_exist:
                os.makedirs(path+"/crop_AB/"+sub_dir)

        if supervised_crop:
            is_exist = os.path.exists(
                path+"/crop_SC")
            if not is_exist:
                os.makedirs(path+"/crop_SC")
            # dog or cat
            for sub_dir in sub_dir_list:
                is_exist = os.path.exists(
                    path+"/crop_SC/"+sub_dir)
                if not is_exist:
                    os.makedirs(path+"/crop_SC/"+sub_dir)

    def crop(self, path, model, active_layer, weight_layer, supervised_crop=False, augment_amount=None):
        print("--->Start cropping")
        start = time.clock()

        self.makedirs(path, supervised_crop)

        if self.cams is None:
            self.cams = visul.CAMs(
                imgs_white=self.imgs_white,
                model=model,
                active_layer='conv_pw_13_relu',
                weight_layer='conv_preds')

        process_bar = io.ShowProcess(len(self.labels_origin))
        for i in range(len(self.labels_origin)):
            process_bar.show_process()

            img_show = self.imgs_origin[i]

            xAB, yAB, wAB, hAB, xSC, ySC, xxSC, yySC = visul.cropMask(
                self.cams[i], img_show)

            img_crop = img_show[yAB:yAB+hAB, xAB:xAB+wAB]
            img_crop = Image.fromarray(img_crop)
            img_crop.save(path+"/crop_AB/" +
                          self.labels_origin[i]+"/"+self.names_origin[i])

            if supervised_crop:
                for j in range(augment_amount):
                    xx = randint(xSC, xxSC)
                    yy = randint(ySC, yySC)
                    imgs_SC = img_show[yy:yy+hAB, xx:xx+wAB]
                    imgs_SC = Image.fromarray(imgs_SC)
                    imgs_SC.save(
                        path+"/crop_SC/"+self.labels_origin[i]+"/"+self.names_origin[i].split('.')[0] + "_"+str(j)+".jpg")

        end = time.clock()
        print("")
        print("Cost time: %.3fs" % (end-start))
        print("")
