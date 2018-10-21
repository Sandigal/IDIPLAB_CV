# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagle

分别读取每个类下的所有文件，用flow对每类图片分别处理
输出图像分类到对应类文件夹内
batch_size<图片总数时，无法保证每张图片数量相同


"""

import numpy as np
from PIL import Image

import dataset_io as io

# %%

dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
        path="../data/FOB 224x224",
        shape=(224, 224))
dataset.create_h5("FOB_train.h5")

#%%

dataset = io.Dataset(augment=True)
class_to_index, sample_per_class = dataset.load_data(
        path="dataset_1_1_origin.h5",
        shape=(224, 224))

# %%

name = "dataset_test.h5"
dataset.create_h5(name)

# %%

dataset.load_h5(name)

# %%

_, _, imgs_test, labels_test = dataset.train_test_split(
    test_shape=0.2)

# %%

imgs_train, labels_train, imgs_valid, labels_valid, names_valid = dataset.cross_split(
    total_splits=4, valid_split=0)

#%%

#from sklearn.datasets import make_classification
#X = imgs_valid.reshape((imgs_valid.shape[0],-1))
#labels_valid = io.label_str2index(labels_valid, class_to_index)
#y = labels_valid
#
#
#from imblearn.over_sampling import SMOTE, ADASYN
#X_resampled, y_resampled = SMOTE().fit_sample(X, y)
#X_resampled, y_resampled = ADASYN().fit_sample(X, y)
#
#from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_sample(X, y)
#
#from imblearn.combine import SMOTEENN
#smote_enn = SMOTEENN(random_state=0)
#X_resampled, y_resampled = smote_enn.fit_sample(X, y)
#
#X_resampled=X_resampled.astype("uint8")
#imgs_valid_resampled=X_resampled.reshape((-1,imgs_valid.shape[1],imgs_valid.shape[2],imgs_valid.shape[3]))
#
#import visul
#grid = visul.show_grid(imgs_valid_resampled[71:])
#grid = visul.show_grid(imgs_valid[48:])

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
