# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:30:43 2018

@author: Sandiagal
"""

import numpy as np
import dataset_io as io
import metrics

from pickle import load
import matplotlib.pyplot as plt
from scipy import interp


# %%

dataset = io.Dataset(path="../data/dataset 336x224/")
class_to_index, sample_per_class = dataset.load_data()

_, _, imgs_test, labels_test = dataset.train_test_split(test_shape=0.2)

imgs_train, labels_train, imgs_valid, labels_valid = dataset.cross_split(
        total_splits=3, valid_split=0)

labels_train = io.label_str2index(labels_train, class_to_index)
labels_valid = io.label_str2index(labels_valid, class_to_index)
labels_valid = io.to_categorical(labels_valid, len(class_to_index))

#mean, std = load(open("mean-std.json", "rb"))
#imgs_train = (imgs_train-mean)/std
#imgs_valid_white = (imgs_valid-mean)/std
#imgs_test = (imgs_test-mean)/std

del dataset

# Class index: {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4, "N1": 5, "N2": 6, "N3": 7, "N4": 8, "N5": 9, "N6": 10, "P1": 11, "P2": 12, "P3": 13, "P4": 14, "P5": 15}
# Sample per class: {"C5": 95, "N4": 181, "P2": 2, "C1": 220, "N2": 83, "P5": 20, "N1": 24, "C4": 66, "P4": 18, "N6": 89, "P1": 6, "C2": 60, "N3": 281, "P3": 6, "C3": 120, "N5": 83}

# %%

# with CustomObjectScope({"relu6": relu6}):
#    model = load_model("2018-07-03 model.h5")
#
#batch_size = 32
# score_predict = model.predict(
#    imgs_valid,
#    batch_size=batch_size,
#    verbose=1)

f = open("20180704_validSplit.1_result.h5", "rb")
contact = load(f)
mean = contact["mean"]
std = contact["std"]
score_predict = contact["score_predict"]
f.close()

#score_predict_max = np.max(score_predict, axis=1)
#label_predict = np.argmax(score_predict, axis=1)
# print("Predicted:", decode_predictions(
#    score_predict, io.reverse_dict(class_to_index), top=3))

# %%

#label_fix = {}
#label_fix[7] = 0
#label_fix[5] = 1
#label_fix[12] = 2
#label_fix[4] = 3
#label_fix[9] = 4
#label_fix[11] = 5
#label_fix[1] = 6
#label_fix[0] = 7
#label_fix[6] = 8
#label_fix[3] = 9
#label_fix[13] = 10
#label_fix[14] = 14
#label_fix[15] = 15
#
#label_fix[2] = 11
#label_fix[8] = 13
#label_fix[10] = 12
#
#anti_label_fix = io.reverse_dict(label_fix)
#
#label_predict = np.argmax(score_predict, axis=1)
#label_predict = io.label_str2index(label_predict, label_fix)
#print("After Fix - predict_generator:", np.mean(label_predict == labels_valid))

# %%

#has = [5, 6, 7, 8, 9, 10]
#a = labels_valid == ban[0]
#for i in range(1, len(ban)):
#    a |= labels_valid == ban[i]
#
#score_predict_max = score_predict_max[~a]
#labels_valid = labels_valid[~a]
#label_predict = label_predict[~a]

# %%

metrics.worst_samples(imgs_valid, labels_valid, score_predict, top=16)

#%%



#%%




# %%

print(metrics.classification_report(
    labels_valid,
    label_predict,
    target_names=np.array(list(class_to_index.keys()))[has]))

# %%

from itertools import product


def plot_confusion_matrix(cm, class_to_index,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
        #        print("Confusion matrix, without normalization")

        #    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_to_index))
    plt.xticks(tick_marks, class_to_index, rotation=45)
    plt.yticks(tick_marks, class_to_index)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


cnf_matrix = metrics.confusion_matrix(labels_valid, label_predict)
plt.figure(figsize=(13, 7))
plt.subplot(121)
plot_confusion_matrix(
    cnf_matrix,
    class_to_index=np.array(list(class_to_index.keys()))[has],
    title="Confusion matrix, without normalization")
plt.subplot(122)
plot_confusion_matrix(
    cnf_matrix,
    class_to_index=np.array(list(class_to_index.keys()))[has],
    normalize=True,
    title="Normalized confusion matrix")
plt.show()
plt.savefig("Confusion matrix")

# %%


def weighted_std(values, weights, axis=None):
    average = np.average(values, weights=weights, axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return np.sqrt(variance)


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(15, 9))
plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r",
         label="Luck", alpha=.8)

#a=labels_valid ==has[0]
# for i in range(1,len(has)):
#    print(i)
#    a=a | (labels_valid ==has[i])

for i in has:
    a = labels_valid == i
    b = label_predict == i
    fpr, tpr, thresholds = metrics.roc_curve(
        labels_valid[a | b],
        score_predict[a | b][:, i],
        pos_label=i)
    fpr[np.isnan(fpr)] = 0.
    fpr[-1] = 1.
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label="ROC for %s (AUC = %0.2f)" % (list(class_to_index.keys())[i], roc_auc))

#mean_tpr = np.average(tprs, weights=vldPerClass, axis=0)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
#std_auc = weighted_std(aucs, vldPerClass)
std_auc = np.std(aucs, axis=0)
plt.plot(mean_fpr, mean_tpr, color="b",
         label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
         lw=2, alpha=.8)

#std_tpr = weighted_std(tprs, vldPerClass, axis=0)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.2,
                 label=r"$\pm$ 1 std. dev.")

fig = plt.gcf()
fig.subplots_adjust(right=.65)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic curve")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           fontsize=14, shadow=1)
plt.show()
plt.savefig("Receiver operating characteristic curve")

# %%

precision = []
recall = []
precision_recall_auc = []
precisions = []

plt.figure(figsize=(15, 9))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    if f_score == 0.2:
        plt.plot(x[y >= 0], y[y >= 0], color="gray",
                 alpha=0.6, label="Iso-f1 curves")
    else:
        plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.6)
    plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + .02))

mean_recall = np.linspace(0, 1, 100)
for i in has:
    a = labels_valid == i
    b = label_predict == i
    P, R, _ = metrics.precision_recall_curve(
        labels_valid[a | b],
        score_predict[a | b][:, i],
        pos_label=i)
#    if len(R) > 2:
#        P = P[:np.argmin(R)]
#        P[-1] = 1.
#        R = R[:np.argmin(R)]
#        R[np.isnan(R)] = 1.
    precision.append(P)
    recall.append(R)
    precision_recall_auc.append(metrics.auc(R, P))
    precisions.append(
        interp(mean_recall, np.flip(R, axis=0), np.flip(P, axis=0)))
    plt.plot(R, P, lw=1, alpha=.3, label="PR for {0} (AUC = {1:0.2f})".format(
        list(class_to_index.keys())[i], metrics.auc(R, P)))

#mean_precision = np.average(precisions, weights=vldPerClass, axis=0)
mean_precision = np.mean(precisions, axis=0)
mean_precision_recall_auc = metrics.auc(mean_recall, mean_precision)
#std_precision_recall_auc = weighted_std(precision_recall_auc, vldPerClass)
std_precision_recall_auc = np.std(precision_recall_auc)
plt.plot(mean_recall, mean_precision, color="b", lw=2, alpha=.8,
         label=r"Mean PR (AUC = {0:0.2f} $\pm$ {1:0.2f})".format(mean_precision_recall_auc, std_precision_recall_auc))

#std_precisions = weighted_std(precisions, vldPerClass, axis=0)
std_precisions = np.std(precisions, axis=0)
precision_upper = np.minimum(mean_precision + std_precisions, 1)
precision_lower = np.maximum(mean_precision - std_precisions, 0)
plt.fill_between(mean_recall, precision_lower, precision_upper,
                 color="grey", alpha=.2, label=r"$\pm$ 1 std. dev.")

fig = plt.gcf()
fig.subplots_adjust(right=.65)
plt.xlim([-.05, 1.05])
plt.ylim([-.05, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           fontsize=14, shadow=1)
plt.show()
plt.savefig("Precision-Recall curve")
