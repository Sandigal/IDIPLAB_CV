# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:30:43 2018

@author: Sandiagal
"""

from pickle import load

from keras.applications.mobilenet import relu6
from keras.models import load_model
import matplotlib.pyplot as plt
import PIL.Image as Image

import dataset_io as io
import visul

# %%

dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
        path="dataset_1_1_origin.h5",
        shape=(336, 224))

imgs_train, labels_train, imgs_valid, labels_valid = dataset.train_test_split(test_shape=0.5)

labels_train = io.label_str2index(labels_train, class_to_index)
labels_valid = io.label_str2index(labels_valid, class_to_index)

f = open("20180925_result.h5", "rb")
contact = load(f)
mean = contact["mean"]
std = contact["std"]
f.close()
imgs_white = (imgs_valid-mean)/std

# %%

visul.show_grid(imgs_train[0:25])

# %%

model = load_model('20180925_model.h5', custom_objects={
                'relu6': relu6})
model.summary()

# %%

index = 206
img_white = imgs_white[index]
img_show = imgs_valid[index].astype(np.uint8)
label_show = labels_valid[index]
feature_layer = 'conv_pw_13_relu'
weight_layer = 'predictions'

cam, mix = visul.CAM(
    img_white=img_white,
    model=model,
    feature_layer=feature_layer,
    weight_layer=weight_layer,
    display=True,
    img_show=img_show,
    label_show=label_show,
    class_to_index=class_to_index,
    top2=False)

xAB, yAB, wAB, hAB, xSC, ySC, xxSC, yySC = visul.cropMask(
        cam=cam,
        img_show=mix,
        display=True)

# %%

plt.ioff()
process_bar = io.ShowProcess(len(imgs_valid))
for i in range(len(imgs_valid)):
    process_bar.show_process()

    index = i
    img_white = imgs_white[index]
    img_show = imgs_valid[index]
    label_show = labels_valid[index]
    feature_layer = 'conv_pw_13_relu'
    weight_layer = 'conv_preds'

    cam, mix = visul.CAM(
        img_white=img_white,
        model=model,
        feature_layer=feature_layer,
        weight_layer=weight_layer,
        display=True,
        img_show=img_show,
        label_show=label_show,
        class_to_index=class_to_index,
        top2=True)


# %%

imgs = []
for i in range(0, len(labels_origin), len(labels_origin)//9):
    print(i)
    index = i
    img_white = imgs_white[index]
    _, mix = visul.CAM(img_white=img_white, model=model, feature_layer='conv_pw_13_relu',
                       weight_layer='conv_preds')
    mix = Image.fromarray(mix)
    imgs.append(mix)

grid = visul.show_grid(imgs)
# grid.save("25.jpg")
plt.imshow(grid)


# %%

path = "E:/Temporary/Deap Leraning/Medical AI/训练结果/7.19 xceptionGAP/20180719_result.h5"
visul.show_history(path)

# %%


