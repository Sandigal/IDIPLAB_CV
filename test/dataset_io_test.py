# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagle

分别读取每个类下的所有文件，用flow对每类图片分别处理
输出图像分类到对应类文件夹内
batch_size<图片总数时，无法保证每张图片数量相同


"""

import dataset_io as io
from PIL import Image
import numpy as np

# %%

path = "../data/dataset 336x224"
augment = False
shape = (336, 224)

dataset = io.Dataset(path, shape=shape, augment=augment)

class_to_index, sample_per_class = dataset.load_data()

del path, augment, shape

# %%

name = "dataset_1_1_origin.h5"
dataset.create_h5(name)

# %%

dataset.load_h5(name)

# %%

test_shape = 0.2
_, _, imgs_test, labels_test = dataset.train_test_split(
    test_shape=test_shape)

del test_shape

# %%

total_splits = 3
valid_split = 0
imgs_train, labels_train, imgs_valid, labels_valid = dataset.cross_split(
    total_splits=total_splits, valid_split=valid_split)

del total_splits, valid_split

# names_train = np.array(dataset.names_origin)[
#    dataset.train_index][dataset.train_cross_index]
#names_train = names_train.repeat(2, axis=0)
# names_augment = np.array(dataset.names_augment)[
#    dataset.augment_index][dataset.augment_cross_index]

# %%
#
#imgs_train, labels_train=io.shuffle(imgs_train, labels_train)
#
# '''
# 数据分割 训练集-验证集-测试集
# '''
#
#
# '''
# 将标签中字符映射为序号
# '''
labels_train = io.label_str2index(labels_train, class_to_index)
labels_valid = io.label_str2index(labels_valid, class_to_index)
labels_test = io.label_str2index(labels_valid, class_to_index)
#
#
labels_train = io.to_categorical(labels_train, len(class_to_index))


labels_train2 = io.label_smooth(labels_train, [0, 5, 11, 16])
