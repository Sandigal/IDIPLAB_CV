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

import dataset_io as io
import visul

from random import randint
import os
import numpy as np
import time
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator


# %%


class AugmentGenerator(object):

    def __init__(self,
                 path,
                 shape=None):
        self.path = path
        self.shape = shape

    def normol_augment(self,datagen_args,augment_amount):
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

    def supervisd_augment(self,datagen_args,augment_amount):

        print("--->Start augmentation")

        datagen = ImageDataGenerator(**datagen_args)

        start = time.clock()

        is_exist = os.path.exists(
            self.path+"/crop_SC_augment")
        if not is_exist:
            os.makedirs(self.path+"/crop_SC_augment")

        # dog or cat
        sub_dir_list = os.listdir(self.path+"/crop_SC")
        process_bar = io.ShowProcess(len(sub_dir_list))
        for sub_dir in sub_dir_list:
            process_bar.show_process(sub_dir)

            is_exist = os.path.exists(
                self.path+"/crop_SC_augment/"+sub_dir)
            if not is_exist:
                os.makedirs(self.path+"/crop_SC_augment/"+sub_dir)

            imgs, names = io.read_imgs_in_dir(
                self.path+"/crop_SC/"+sub_dir, self.shape)
            imgs = np.array(imgs)

            # dog1 or dog2
            for img, name in zip(imgs, names):
                img = np.expand_dims(img, axis=0)

                augmentgen = datagen.flow(
                    img,
                    batch_size=1,
                    shuffle=False,
                    save_to_dir=self.path+"/crop_SC_augment/"+sub_dir,
                    save_format='jpg')

                for i in range(augment_amount):
                    augmentgen.save_prefix = name.split(
                        '.')[0] + "_"+str(np.random.randint(0, 99999))
                    augmentgen.next()

        end = time.clock()
        print("")
        print("Cost time:", end-start, "s")
        print("")



# %%


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
