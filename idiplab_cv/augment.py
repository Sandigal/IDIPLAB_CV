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
from keras.preprocessing.image import ImageDataGenerator
import glob
import os
import numpy as np
import time
from PIL import Image
import visul

# %%


class AugmentGenerator(object):

    def __init__(self,
                 path,
                 datagen_args,
                 augment_amount=10,
                 shape=None):
        self.path = path
        self.datagen_args = datagen_args
        self.augment_amount = augment_amount
        self.shape = shape

    def change(self, **kw):
        self.repeat_time = 0
        self.generator_index += 1
        if 'path' in kw:
            self.path = kw['path']
        if 'datagen_args' in kw:
            self.datagen_args = kw['datagen_args']
        if 'augment_amount' in kw:
            self.augment_amount = kw['augment_amount']
        if 'shape' in kw:
            self.shape = kw['shape']

    def creat1(self):
        while True:
            print("--->Start augmentation")

            datagen = ImageDataGenerator(**self.datagen_args)

            start = time.clock()

            is_exist = os.path.exists(
            self.path+"/augment")
            if not is_exist:
                os.makedirs(self.path+"/augment")

            # dog or cat
            sub_dir_list = os.listdir(self.path+"/origin")
            for sub_dir in sub_dir_list:

                is_exist = os.path.exists(
                    self.path+"/augment/"+sub_dir)
                if not is_exist:
                    os.makedirs(self.path+"/augment/"+sub_dir)

                imgs, names = io.read_imgs_in_dir(
                    self.path+"/origin/"+sub_dir, self.shape)
                imgs = np.array(imgs)

                # dog1 or dog2
                for img, name in zip(imgs, names):
                    img = np.expand_dims(img, axis=0)

                    augmentgen = datagen.flow(
                        img,
                        batch_size=1,
                        shuffle=False,
                        save_to_dir=self.path+"/augment/"+sub_dir,
                        save_format='jpg')

                    for i in range(self.augment_amount):
                        augmentgen.save_prefix = name.split(
                            '.')[0] + "_"+str(np.random.randint(0, 99999))
                        augmentgen.next()

            end = time.clock()
            print("Cost time:", end-start, "s")
            print("")

            yield None

    def creat2(self):
        while True:
            print("--->Start augmentation")

            path_agument = self.path
            path_agument_suffixes = '_augment'
            datagen = ImageDataGenerator(**self.datagen_args)

            start = time.clock()

            g = os.walk(self.path)

            for path, dir_list, file_list in g:
                # dir_list:类别的list，例如['cats', 'dogs']
                for dir_name in dir_list:
                    # dir_name:类别的名称，例如cats

                    # 跳过_augment文件夹
                    if dir_name.rfind("_augment") != -1:
                        continue

                    # 生成_augment文件夹，确定增强批次
                    is_exists = os.path.exists(
                        path_agument+dir_name+path_agument_suffixes)
                    if not is_exists:
                        os.makedirs(path_agument+dir_name +
                                    path_agument_suffixes)

                    imgs = []
                    for filename in glob.glob(self.path+dir_name+'/*.jpg'):
                        #  filename:带有路径的图片完整地址，例如../Data_Origin/cats\rezero_icon_10.jpg

                        img = Image.open(filename)
                        if self.shape is not None:
                            img = img.resize(self.shape)
                        img = np.asarray(img)
                        imgs.append(img)

                    batch_size = 32
                    imgs = np.array(imgs)
                    augmentgen = datagen.flow(
                        imgs,
                        batch_size=batch_size,
                        shuffle=True,
                        save_to_dir=path_agument+dir_name+path_agument_suffixes,
                        save_format='jpg')

                    for i in range(self.augment_amount):
                        batches = 0
                        for _ in augmentgen:
                            augmentgen.save_prefix = str(
                                np.random.randint(0, 99999))
                            batches += 1
                            if batches >= len(imgs) / batch_size:
                                break
#                    augmentgen.next()

                break

            end = time.clock()
            print("Cost time:", end-start, "s")
            print("")

            yield None

    def next(self, Multithreading=False):
        if Multithreading:
            next(self.creat2())
        else:
            next(self.creat1())

# %%

class cropGenerator(object):

    def __init__(self,
                 imgs_origin,
                 imgs_white,
                 labels_origin,
                 names_origin
                 ):
        self.labels_origin = labels_origin
        self.imgs_white = imgs_white
        self.imgs_white = imgs_white
        self.names_origin=names_origin

    def crop(self,path,mean_std,model,active_layer, weight_layer ):
        print("--->Start cropping")
        start = time.clock()

        is_exist = os.path.exists(
            path+"/crop")
        if not is_exist:
            os.makedirs(path+"/crop")

        # dog or cat
        sub_dir_list = os.listdir(path+"/origin")
        for sub_dir in sub_dir_list:
            is_exist = os.path.exists(
                path+"/crop/"+sub_dir)
            if not is_exist:
                os.makedirs(self.path+"/crop/"+sub_dir)

        for i in range(len(self.labels_origin)):
            print(i)
            target = self.imgs_origin[i]
            target2 = self.imgs_white[i]
            out, cam = visul.CAM(img=target, img2=target2, model=model, finalActiveLayerName='conv_pw_13_relu',
                                 weightLayerName='conv_preds')
            xAB, yAB, wAB, wAB, xSC, ySC, xxSC, yySC = visul.cropMask(cam, out)

            target3 = target[yAB:yAB+wAB, xAB:xAB+wAB]
            target3 = Image.fromarray(target3)
            target3.save(path+"/crop/"+self.labels_origin[i]+"/"+self.names_origin[i])

        end = time.clock()
        print("Cost time:", end-start, "s")
        print("")
