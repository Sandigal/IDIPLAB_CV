# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagle

测试idiplab_cv.dataset_io的各种API

"""

import idiplab_cv.dataset_io as io

# %%

# 创建数据集对象
dataset = io.Dataset(augment=False)

# 读取指定路径下图像
class_to_index, sample_per_class = dataset.load_data(
        path="data/nuclear_去模糊_349_crop_86x178",
        shape=(86, 178))

# 散装图像打包
dataset.create_h5("nuclear_deblur_crop.h5")

#%% 读取打包好的数据集

dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
        path="dataset_1_1_origin.h5")

#%%

# 划分训练-测试集
imgs_train, labels_train, imgs_valid, labels_valid = dataset.train_test_split(
    total_splits=5, test_split=4)

# 产生VOC格式的训练、测试文件索引
dataset.VOC_Indexes()

# 划分训练-验证集
imgs_train, labels_train, imgs_valid, labels_valid, names_valid = dataset.cross_split(
    total_splits=4, valid_split=0)

# 将标签中字符映射为序号
labels_train = io.label_str2index(labels_train, class_to_index)

# 将序号转为one-hot编码
labels_train = io.to_categorical(labels_train, len(class_to_index))

# 标签平滑
labels_train = io.label_smooth(labels_train, [0, 6])
