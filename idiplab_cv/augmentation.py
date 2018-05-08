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

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import glob
import os
import numpy as np
from skimage import transform
from preprocessing_funciton import noise
import time
from PIL import Image

# %%


# %%

def augmentation(path_origin, agument_time, new_shape, data_gen_args):

    print("--->Start augmentation")

    path_agument = path_origin
    path_agument_suffixes = '_agmt'

    start = time.clock()

    train_datagen = ImageDataGenerator(**data_gen_args)

    g = os.walk(path_origin)

    for path, dir_list, file_list in g:
        # dir_list:类别的list，例如['cats', 'dogs']
        for dir_name in dir_list:
            # dir_name:类别的名称，例如cats

            is_exists = os.path.exists(
                path_agument+dir_name+path_agument_suffixes)
            if not is_exists:
                os.makedirs(path_agument+dir_name+path_agument_suffixes)

            for filename in glob.glob(path_origin+dir_name+'/*.jpg'):
                #  filename:带有路径的图片完整地址，例如../Data_Origin/cats\rezero_icon_10.jpg
                img = Image.open(filename)  # 这是一个PIL图像
                img = np.asarray(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
    #            x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)

                # 设置图像尺寸
                height, width = img.shape[:2]
                x = int((width-height)/2)
                w = height
                img = img[:, x:w+x, :]
                img = transform.resize(img/255, new_shape, mode='constant')*255
                # this is a Numpy array with shape (1, 3, 150, 150)
                img = np.expand_dims(img, axis=0)
                img=np.tile(img,(agument_time,1,1,1))

                # 设置生成流
                train_datagen.fit(img)
                train_generator = train_datagen.flow(
                    img,
                    batch_size=agument_time,
                    shuffle=True,
                    save_prefix=filename.split('\\')[-1].split('.')[0],
                    save_to_dir=path_agument+dir_name+path_agument_suffixes,
                    save_format='jpg')

                # 循环生成
                for i in range(int(1)):
#                    cur_img=img
#                    train_generator.x=cur_img
#                    train_generator.save_prefix=filename.split('\\')[-1].split('.')[0]+"_"+str(i)
                    train_generator.next()

        break
    end = time.clock()
    print("Cost time:", end-start, "s")
    print("")
