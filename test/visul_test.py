# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:30:43 2018

@author: Sandiagal

测试idiplab_cv.visul的各种API

"""

from pickle import load

import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

import idiplab_cv.dataset_io as io
from idiplab_cv import visul

# %% 读取数据

# 原始数据
dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
    path="nuclear_deblur_336x224.h5",
    shape=(336, 224))
imgs_train, labels_train, imgs_test, labels_test = dataset.train_test_split(
    total_splits=4, test_split=3)
labels_train = io.label_str2index(labels_train, class_to_index)

# 训练数据
f = open("Record/20190107_0_result.h5", "rb")
contact = load(f)
EPOCHS = contact["epochs"],
history = contact["history"]
labels_test = contact["labels_test"]
mean = contact["mean"]
#names_valid2 = contact["names_valid"]
scores_predict = contact["scores_predict"]
std = contact["std"]
f.close()
imgs_white = (imgs_test-mean)/std

# 特征图
f = open("Record/20190107_0_feature_maps.h5", "rb")
contact = load(f)
feature_maps = contact["feature_maps"]
# feature=np.mean(feature,axis=(1,2))
weights = contact["weights"]
f.close()

# %%

# 栅格展示图像数据
visul.show_grid(imgs_train[0:25])

# 智能筛选出部分图像展示
visul.overall(imgs_train)

# 栅格展示分类效果最差的样本
visul.worst_samples(imgs_test, labels_test,
                      scores_predict, class_to_index, top=16, names_valid=None)

# %% 单张图CAM、以知特征图时可以快速运算

# 图像序号
index = 84

# 计算单张图CAM
cam, mix = visul.CAM(
    feature_map=np.expand_dims(feature_maps[index], axis=0),
    weights=weights,
    scores_predict=scores_predict[index],
    display=True,
    img_show=imgs_test[index].astype(np.uint8),
    class_to_index=class_to_index,
    label_show=labels_test[index]
)

# 计算监督裁剪的ABSC
xAB, yAB, wAB, hAB, xSC, ySC, xxSC, yySC = visul.cropMask(
    cam=cam,
    img_show=mix,
    display=True)


# %% 计算所有样本的CAM，显示出原始图，top1，top2，true的CAM结果。未知特征图，需要额外求解

from keras.models import load_model
model = load_model('Record/20190107_0_model.h5')
# model.summary()

feature_layer = 'activation_98'
weight_layer = 'predictions'
plt.ioff()
process_bar = io.ShowProcess(len(imgs_test))
for index in range(len(labels_test)):
    process_bar.show_process()

    img_white = imgs_white[index]
    img_show = imgs_test[index]
    label_show = labels_test[index]

    cam, mix = visul.CAM(
        img_white=img_white,
        model=model,
        feature_layer=feature_layer,
        weight_layer=weight_layer,
        display=True,
        img_show=img_show.astype(np.uint8),
        label_show=label_show,
        class_to_index=class_to_index,
        extend=True)
plt.ion()

# %% 并行计算所有样本的CAM

feature_layer = 'activation_98'
weight_layer = 'predictions'
cams = visul.CAMs(
    imgs_white=imgs_white[:32],
    model=model,
    feature_layer=feature_layer,
    weight_layer=weight_layer)

# %% 模型分级结果和对应的CAM的示例

# 原始图像
dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
    path="nuclear_deblur_336x224.h5",
    shape=(336, 224))
_, _, imgs_test, _ = dataset.train_test_split(
    total_splits=4, test_split=3)

# 预测置信度
f = open("Record/20190107_0_result.h5", "rb")
contact = load(f)
labels_test = contact["labels_test"]
scores_predict = contact["scores_predict"]
f.close()

# 特征图
f = open("Record/20190107_0_feature_maps.h5", "rb")
contact = load(f)
feature_maps = contact["feature_maps"]
weights = contact["weights"]
f.close()

# 计算综合CAM图
feature_layer = 'activation_98'
weight_layer = 'predictions'
visul.best_worst_samples(
    imgs_valid=imgs_test,
    labels_valid=labels_test,
    feature_maps=feature_maps,
    weights=weights,
    scores_predict=scores_predict,
    class_to_index=class_to_index)
