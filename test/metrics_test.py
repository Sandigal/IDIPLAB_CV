# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:30:43 2018

@author: Sandiagal
"""

from pickle import load

import matplotlib.pyplot as plt
import numpy as np

import idiplab_cv.dataset_io as io
from idiplab_cv import metrics

# %%

dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
    path="nuclear_deblur_336x224.h5",
    shape=(1, 1))

_, _, imgs_test, labels_test = dataset.train_test_split(
    total_splits=4, test_split=3)
imgs_train, labels_train, imgs_valid, labels_valid, names_valid = dataset.cross_split(
    total_splits=3, valid_split=0)

labels_test = io.label_str2index(labels_test, class_to_index)


# %% 读取训练数据

f = open("Record/20190107_0_result.h5", "rb")
contact = load(f)
test_acc = contact["test_acc"]
class_to_index = contact["class_to_index"]
epoches_min = contact["epoches_min"]
EPOCHS = contact["epochs"]
history = contact["history"]
labels_valid = contact["labels_test"]
mean = contact["mean"]
#print('{:.32f} {:.32f} {:.32f}'.format(mean[0][0][0][0], mean[0][0][0][1], mean[0][0][0][2]))
scores_predict = contact["scores_predict"]
std = contact["std"]
#print('{:.32f} {:.32f} {:.32f}'.format(std[0][0][0][0], std[0][0][0][1], std[0][0][0][2]))
f.close()

class_to_index = dict(sorted(class_to_index.items(),
                             key=lambda item: item[1], reverse=False))
labels_predict = np.argmax(scores_predict, axis=1)

# %% 单模型评价系统

# 单次训练下的训练曲线。一个图表示训练测试的损失，另一个表示训练测试的准确率。
metrics.show_history_pair(history, EPOCHS=EPOCHS)

# 单次训练下的训练曲线。画出每个阶段时测试集的准确率
metrics.show_history_section(history=history, test_acc=test_acc,
                             title="learn curve", EPOCHS=EPOCHS)

# 基本分类指标
report_dict = metrics.classification_report(
    labels_valid, labels_predict, class_to_index)

# 箱型图
metrics.violinBox(labels_valid, labels_predict, class_to_index, section=None)

# ROC曲线
auc = metrics.ROC(labels_valid, scores_predict, class_to_index)

# PR曲线
AP = metrics.PR(labels_valid, scores_predict, class_to_index, section=[0, 6])

# 上述API汇总
metrics.summary_simple(
    labels_valid, labels_predict, scores_predict, class_to_index)

# 用于病情分级、治疗建议的API汇总
ee, r0, r1, mAP, F1, mAPT = metrics.summary(
    labels_valid, labels_predict, scores_predict, class_to_index)

# %% 画出单个模型，多次训练下的训练曲线。

test_accs = []
historys = []
for i in range(3):
    f = open("Record/20190107_"+str(i)+"_result.h5", "rb")
    contact = load(f)
    test_acc = contact["test_acc"]
    EPOCHS = contact["epochs"]
    history = contact["history"]
    f.close()

    test_accs.append(test_acc)
    historys.append(history)

metrics.show_history_cross(historys, test_accs=test_accs, EPOCHS=EPOCHS)


# %% 多模型评价系统

historys1 = []
labels_valid1 = []
scores_predict1 = []
for i in range(3):
    f = open("Record/20190107_"+str(i)+"_result.h5", "rb")
    contact = load(f)
    test_acc = contact["test_acc"]
    class_to_index = contact["class_to_index"]
    epoches_min = contact["epoches_min"]
    EPOCHS = contact["epochs"]
    history = contact["history"]
    labels_valid = contact["labels_test"]
    mean = contact["mean"]
    scores_predict = contact["scores_predict"]
    std = contact["std"]
    f.close()
    historys1.append(history)
    labels_valid1.append(labels_valid)
    scores_predict1.append(scores_predict)

historys2 = []
labels_valid2 = []
scores_predict2 = []
for i in range(3):
    f = open("Record/20190109_"+str(i)+"_result.h5", "rb")
    contact = load(f)
    test_acc = contact["test_acc"]
    class_to_index = contact["class_to_index"]
    epoches_min = contact["epoches_min"]
    EPOCHS = contact["epochs"]
    history = contact["history"]
    labels_valid = contact["labels_test"]
    mean = contact["mean"]
    scores_predict = contact["scores_predict"]
    std = contact["std"]
    f.close()
    historys2.append(history)
    labels_valid2.append(labels_valid)
    scores_predict2.append(scores_predict)

class_to_index = dict(sorted(class_to_index.items(),
                             key=lambda item: item[1], reverse=False))
historyss = [historys1, historys2]
labels_valids = [labels_valid1, labels_valid2]
scores_predicts = [scores_predict1, scores_predict2]
subtitles = ["origin",
             "deblur"]

# 训练曲线
metrics.show_historys_compare(
    historyss=historyss, title="Learning curves if augment or not", subtitles=subtitles)

# ROC曲线
metrics.ROC_compare(labels_valids, scores_predicts,
                    class_to_index, subtitles, section=[0, 5])

# PR曲线
metrics.PR_compare(labels_valids, scores_predicts,
                   class_to_index, subtitles, section=[0, 5])
