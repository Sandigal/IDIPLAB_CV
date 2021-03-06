# -*- coding: utf-8 -*-
"""
该模块 :meth:`augment` 包含数据增强的类和函数。
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0


import os
from random import randint
import time

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

import idiplab_cv.dataset_io as io
from idiplab_cv import visul


class AugmentGenerator(object):
    """数据增强生成器。

    当实例化 :meth:`AugmentGenerator` 时，您需要指出数据所在目录地址和图像的输出尺寸。尽管部分模型支持多尺度输入，但同一批训练数据必须保持相同维度。您可以分批产生多尺度数据，再分批训练。

    Args:
        path (:obj:`str`): 数据所在目录地址。目录结构请参照 :ref:`目录结构`。
        shape (:obj:`turple` of :obj:`int`, 可选): 所有的图像将以该尺寸来输出。格式为 `(width, height, channels)`，默认为 `(336, 224, 3)`。
    """

    def __init__(self,
                 path,
                 shape=None):
        self.path = path
        self.shape = shape

    def normol_augment(self, datagen_args, augment_amount=10):
        """非监督数据增强。

         :meth:`normol_augment` 将从指定目录下的 `origin` 文件夹中读取所有数据，并创建 `augment` 文件夹来存放增强数据。需要目录的具体格式可以参见参照 :ref:`目录结构`。为了保证处理成功，请保证目录内不存在 **Thumbs.db** 等隐藏文件。

         函数将按照 ``datagen_args`` 中设定的方法对所给图像进行 ``augment_amount`` 次数据增强。

        Note:
            :meth:`normol_augment` 实际是调用了 keras的 ImageDataGenerator_ 方法来增强数据。ImageDataGenerator在保存生成的增强数据时会在文件名后添加"_0_%4d"的随机数。当后续生成的随机数与前面的重复时，将会覆盖前一个生成的增强数据，导致实际增广倍数与设定不一。

            这对于 :meth:`dataset_io.Dataset.cross_split`，数据集分割中只提取 **训练集** 部分的增强数据的函数，将会产生致命的错误。对此已经采取了一定缓解措施：另外添加了"%5d"的随机数，这样单个图像覆盖的概率为1e-7%。尽管如此，依旧强烈建议 **手动检查 `augment` 文件夹中是否发生了覆盖现象**。

            .. _ImageDataGenerator: https://github.com/aleju/imgaug

        Args:
            datagen_args (:obj:`dict`): 可选数据增强方法。选用的方法将依次叠加进行处理。所有支持方法的具体介绍可以参照 :ref:`数据增强方法`。
            augment_amount (:obj:`int`, 可选): 数据增广倍数，默认为`10`。

        Examples:

            >>> agmtgen = agmt.AugmentGenerator(path="../data/dataset 336x224")
            >>> datagen_args = dict(
            ...         rotation_range=15.,
            ...         width_shift_range=0.05,
            ...         height_shift_range=0.05)
            >>> agmtgen.normol_augment(datagen_args=datagen_args, augment_amount=2)
            --->Start augmentation
            -->Processing for C1 [=========>. . . . . . . . . . . . . . . . . . . . ] 33.33%
            -->Processing for C2 [===================>. . . . . . . . . . ] 66.67%
            -->Processing for C3 [=============================>] 100.00%
            [=============================>] 100.00%
            Cost time: 30.498 s

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
        process_bar_dir = io.ShowProcess(len(sub_dir_list))
        for sub_dir in sub_dir_list:
            process_bar_dir.show_process(sub_dir)

            is_exist = os.path.exists(
                self.path+"/augment/"+sub_dir)
            if not is_exist:
                os.makedirs(self.path+"/augment/"+sub_dir)

            imgs, names = io._read_imgs_in_dir(
                self.path+"/origin/"+sub_dir, self.shape)
            imgs = np.array(imgs)

            # dog1 or dog2
            print("")
            process_bar_pic = io.ShowProcess(len(names))
            for img, name in zip(imgs, names):
                process_bar_pic.show_process()
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
        print("Cost time: %.3fs" % (end-start))
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

    def makedirs(self, path, SC_crop=False):
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

        if SC_crop:
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

    def AB_crop(self, path, model, feature_layer, weight_layer, SC_crop=False, augment_amount=None):
        print("--->Start cropping")
        start = time.clock()

        self.makedirs(path, SC_crop)

        if self.cams is None:
            print("计算CAM中，请稍后")
            self.cams = visul.CAMs(
                imgs_white=self.imgs_white,
                model=model,
                feature_layer=feature_layer,
                weight_layer=weight_layer)
        else:
            print("已存在CAM，开始裁剪")

        process_bar = io.ShowProcess(len(self.labels_origin))
        for i in range(len(self.labels_origin)):
            process_bar.show_process()

            img_show = self.imgs_origin[i]

            xAB, yAB, wAB, hAB, xSC, ySC, xxSC, yySC = visul.cropMask(
                self.cams[i], img_show, display=False)

            img_crop = img_show[yAB:yAB+hAB, xAB:xAB+wAB]
            img_crop = Image.fromarray(img_crop)
            img_crop.save(path+"/crop_AB/" +
                          self.labels_origin[i]+"/"+self.names_origin[i])

            if SC_crop:
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
