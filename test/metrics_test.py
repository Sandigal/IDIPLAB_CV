# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:30:43 2018

@author: Sandiagal
"""

from pickle import load

import numpy as np

import dataset_io as io
import metrics

# %%

dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
        path="dataset_1_1_origin.h5",
        shape=(1, 1))

imgs_train, labels_train, imgs_valid, labels_valid = dataset.train_test_split(test_shape=0.2)

imgs_train, labels_train, imgs_valid, labels_valid, names_valid = dataset.cross_split(
    total_splits=3, valid_split=0)

labels_train = io.label_str2index(labels_train, class_to_index)
labels_valid = io.label_str2index(labels_valid, class_to_index)

# Class index: {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4, "N1": 5, "N2": 6, "N3": 7, "N4": 8, "N5": 9, "N6": 10, "P1": 11, "P2": 12, "P3": 13, "P4": 14, "P5": 15}
# Sample per class: {"C5": 95, "N4": 181, "P2": 2, "C1": 220, "N2": 83, "P5": 20, "N1": 24, "C4": 66, "P4": 18, "N6": 89, "P1": 6, "C2": 60, "N3": 281, "P3": 6, "C3": 120, "N5": 83}

# %%

f = open("20180925_result.h5", "rb")
contact = load(f)
#EPOCHS=contact["epochs"],
history = contact["history"]
labels_valid = contact["labels_valid"]
mean = contact["mean"]
#names_valid2 = contact["names_valid"]
scores_predict = contact["scores_predict"]
std = contact["std"]
f.close()

labels_predict = np.argmax(scores_predict, axis=1)

# %%

#from keras.applications.mobilenet import relu6
#
#from keras.models import load_model
#
#
#model = load_model("20180723_model.h5", custom_objects={
#               'relu6': relu6})
#
#imgs_white = (imgs_valid-mean)/std
#score_predict2 = model.predict(imgs_white, verbose=1)


#%%

labels_valids=[]
labels_valids.append(labels_valid)

scores_predicts=[]
scores_predicts.append(scores_predict)

labels_predicts=[]
labels_predicts.append(labels_predict)

subtitles=["Multi step","Weak multi step"]

#%%

labels_valids.append(labels_valid)
scores_predicts.append(scores_predict)
labels_predicts.append(labels_predict)



#%%

metrics.show_history(history=history, EPOCHS=EPOCHS)


# %%

#paths = ["../训练结果/7.9 原数据/",
#         "../训练结果/7.9 增强数据/"]
#subtitles = ["Without Augment",
#             "Contain Augment"]
#metrics.show_cross_historys(
#    paths=paths, title="Learning curves if augment or not", subtitles=subtitles)

#%%

metrics.violinBox(labels_valid, labels_predict, class_to_index,section=[0,5])

#%%

metrics.violinBoxCompare(labels_valids, labels_predicts, class_to_index,subtitles,section=[0,11])

# %%

metrics.worst_samples(imgs_valid, labels_valid,
                      scores_predict, class_to_index, top=16,names_valid=None)

# %%

metrics.classification_report(labels_valid, label_predict, class_to_index)

# %%

metrics.confusion_matrix(labels_valid, label_predict, class_to_index)

# %%

metrics.ROC(labels_valid, scores_predict, class_to_index, section=[0,5,11])

#%%

metrics.ROCCompare(labels_valids, scores_predicts, class_to_index,subtitles, section=[0,5])

# %%

metrics.PR(labels_valid, scores_predict, class_to_index, section=[0,5,11])

#%%

metrics.PRCompare(labels_valids, scores_predicts, class_to_index,subtitles, section=[0,5])



#%%



#%%


