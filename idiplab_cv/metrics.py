# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:30:43 2018

@author: Sandiagal
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import metrics

from dataset_io import reverse_dict
from dataset_io import to_categorical
from visul import show_grid


def _weighted_std(values, weights, axis=None):
    average = np.average(values, weights=weights, axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return np.sqrt(variance)


def decode_predictions(preds, class_index, top=5):
    """
    dsd
    """
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(str(i), class_index[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def worst_samples(imgs_valid, labels_valid, score_predict, class_to_index, top=10):
    """
    dsd
    """
    index_to_class = reverse_dict(class_to_index)
    labels_valid_onehot = to_categorical(labels_valid, len(class_to_index))
    score_valid = np.max(np.array(score_predict)*labels_valid_onehot, axis=1)
    label_predict = np.argmax(score_predict, axis=1)
    score_predict_max = np.max(score_predict, axis=1)
    worst_index = np.argsort(score_predict_max-score_valid)[-top:]

    imgs = []
    suptitles = []
    for _, index in enumerate(worst_index):
        imgs.append(imgs_valid[index])
        suptitles.append(
            "Predict: %s (%5.2f%%)\n True  : %s (%5.2f%%)" % (
                index_to_class[label_predict[index]],
                score_predict_max[index]*100,
                index_to_class[labels_valid[index]],
                score_predict[index][labels_valid[index]]*100))

    plt = show_grid(imgs, "Worst prediction samples", suptitles=suptitles)
    return plt


def number_per_class(labels_valid, class_to_index):
    """
    dsd
    """
    vldPerClass = [np.sum(labels_valid == i) for i in range(len(labels_valid))]
    plt.figure()
    plt.bar(range(len(vldPerClass)), vldPerClass,
            color="rgb", tick_label=np.array(list(class_to_index.keys())))
    plt.title("Number per class")
    plt.savefig("Number for class all")
    return plt


def classification_report(labels_valid, label_predict, class_to_index):
    """
    dsd
    """
    target_names = np.array(list(class_to_index.keys()))
    report=metrics.classification_report(labels_valid, label_predict,
                                        target_names=target_names)
    print(report)
    f=open('classification_report.txt','w')
    f.write(report)
    f.close()

def _plot_confusion_matrix(cm, class_to_index,
                           normalize=False,
                           title="Confusion matrix",
                           cmap=plt.cm.Blues):
    """
    dsd
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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
    plt.ylabel("True")
    plt.xlabel("Predict")
    return plt


def confusion_matrix(labels_valid, label_predict, class_to_index):
    """
    dsd
    """
    cnf_matrix = metrics.confusion_matrix(labels_valid, label_predict)

    plt.figure(figsize=(13, 7))
    plt.subplot(121)
    _plot_confusion_matrix(
        cnf_matrix,
        class_to_index=np.array(list(class_to_index.keys())),
        title="Plain confusion matrix")
    plt.subplot(122)
    _plot_confusion_matrix(
        cnf_matrix,
        class_to_index=np.array(list(class_to_index.keys())),
        normalize=True,
        title="Normalized confusion matrix")

    plt.savefig("Confusion matrix")
    return plt


def ROC(labels_valid, score_predict, class_to_index, section=None):
    """
    dsd
    """
    label_predict = np.argmax(score_predict, axis=1)

    plt.figure(figsize=(15, 9))
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r",
             label="Luck", alpha=.8)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(class_to_index)):

        # ddddddddddddddddddd
        if section is None:
            indexs_sample = labels_valid == i
            tmp = label_predict == i
            indexs_sample = indexs_sample | tmp
        else:
            for j in range(1, len(section)):
                if i < section[j]:
                    indexs_sample = labels_valid == section[j-1]
                    for k in range(section[j-1]+1, section[j]):
                        tmp = labels_valid == k
                        indexs_sample = indexs_sample | tmp
                    break

        fpr, tpr, thresholds = metrics.roc_curve(
            labels_valid[indexs_sample],
            score_predict[indexs_sample][:, i],
            pos_label=i)
        fpr[np.isnan(fpr)] = 0.
        fpr[-1] = 1.
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label="ROC for %s (AUC = %.2f)" % (list(class_to_index.keys())[i], roc_auc))

    colors = ["c", "g", "r", "m", "k"] # c y
    for i in range(1, len(section)):
        mean_tpr = np.mean(tprs[section[i-1]:section[i]], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs[section[i-1]:section[i]], axis=0)
        plt.plot(mean_fpr, mean_tpr, color=colors[i],
                 label=r"Mean ROC for %s (AUC = %.2f $\pm$ %.2f)" % (
                         list(class_to_index.keys())[section[i-1]][0],
                         mean_auc, std_auc),
                 lw=2, alpha=1)

#        std_tpr = np.std(tprs[section[i-1]:section[i]], axis=0)
#        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#        plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
#                         color=colors[i], alpha=.2)

    #mean_tpr = np.average(tprs, weights=vldPerClass, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    #std_auc = weighted_std(aucs, vldPerClass)
    std_auc = np.std(aucs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color="c",
             label=r"Mean ROC (AUC = %.2f $\pm$ %.2f)" % (mean_auc, std_auc),
             lw=2, alpha=.8)

    #std_tpr = weighted_std(tprs, vldPerClass, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                     color="c", alpha=.2,
                     label=r"$\pm$ 1 std. dev.")

    fig = plt.gcf()
    fig.subplots_adjust(right=.65)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
               fontsize=14, shadow=1)
    plt.title("Receiver operating characteristic curve")
    plt.xlabel("False Positive Rate")
    plt.xlim([-0.05, 1.05])
    plt.ylabel("True Positive Rate")
    plt.ylim([-0.05, 1.05])
    plt.savefig("Receiver operating characteristic curve")


def PR(labels_valid, score_predict, class_to_index, section=None):
    """
    dsd
    """
    label_predict = np.argmax(score_predict, axis=1)

    plt.figure(figsize=(15, 9))
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        if f_score == 0.2:
            plt.plot(x[y >= 0], y[y >= 0], color="gray",
                     alpha=0.6, label="Iso-f1 curves")
        else:
            plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.6)
        plt.annotate("f1=%.1f"%f_score, xy=(0.9, y[45] + .02))


    precision = []
    recall = []
    precision_recall_auc = []
    precisions = []
    mean_recall = np.linspace(0, 1, 100)

    for i in range(len(class_to_index)):

        # ddddddddddddddddddd
        if section is None:
            indexs_sample = labels_valid == i
            tmp = label_predict == i
            indexs_sample = indexs_sample | tmp
        else:
            for j in range(1, len(section)):
                if i < section[j]:
                    indexs_sample = labels_valid == section[j-1]
                    for k in range(section[j-1]+1, section[j]):
                        tmp = labels_valid == k
                        indexs_sample = indexs_sample | tmp
                    break

        P, R, _ = metrics.precision_recall_curve(
            labels_valid[indexs_sample],
            score_predict[indexs_sample][:, i],
            pos_label=i)
        precision.append(P)
        recall.append(R)
        precision_recall_auc.append(metrics.auc(R, P))
        precisions.append(
            interp(mean_recall, np.flip(R, axis=0), np.flip(P, axis=0)))
        plt.plot(R, P, lw=1, alpha=.3,
                 label="PR for %s (AUC = %.2f)" % (list(class_to_index.keys())[i], metrics.auc(R, P)))

    colors = ["c", "g", "r", "m", "k"] # c y
    for i in range(1, len(section)):
        mean_precision = np.mean(precisions[section[i-1]:section[i]], axis=0)
        mean_precision_recall_auc = metrics.auc(mean_recall, mean_precision)
        std_precision_recall_auc = np.std(precision_recall_auc[section[i-1]:section[i]], axis=0)
        plt.plot(mean_recall, mean_precision, color=colors[i],
                 label=r"Mean ROC for %s (AUC = %.2f $\pm$ %.2f)" % (
                         list(class_to_index.keys())[section[i-1]][0],
                         mean_precision_recall_auc, std_precision_recall_auc),
                 lw=2, alpha=1)

#        std_precisions = np.std(precisions[section[i-1]:section[i]], axis=0)
#        precision_upper = np.minimum(mean_precision + std_precisions, 1)
#        precision_lower = np.maximum(mean_precision - std_precisions, 0)
#        plt.fill_between(mean_recall, precision_lower, precision_upper,
#                         color=colors[i], alpha=.2)

    #mean_precision = np.average(precisions, weights=vldPerClass, axis=0)
    mean_precision = np.mean(precisions, axis=0)
    mean_precision_recall_auc = metrics.auc(mean_recall, mean_precision)
    #std_precision_recall_auc = weighted_std(precision_recall_auc, vldPerClass)
    std_precision_recall_auc = np.std(precision_recall_auc)
    plt.plot(mean_recall, mean_precision, color="c", lw=2, alpha=.8,
             label=r"Mean PR (AUC = %.2f $\pm$ %.2f)" % (
                     mean_precision_recall_auc, std_precision_recall_auc))

    #std_precisions = weighted_std(precisions, vldPerClass, axis=0)
    std_precisions = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precisions, 1)
    precision_lower = np.maximum(mean_precision - std_precisions, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper,
                     color="c", alpha=.2, label=r"$\pm$ 1 std. dev.")

    fig = plt.gcf()
    fig.subplots_adjust(right=.65)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
               fontsize=14, shadow=1)
    plt.title("Precision-Recall curve")
    plt.xlabel("Recall")
    plt.xlim([-.05, 1.05])
    plt.ylabel("Precision")
    plt.ylim([-.05, 1.05])
    plt.savefig("Precision-Recall curve")
