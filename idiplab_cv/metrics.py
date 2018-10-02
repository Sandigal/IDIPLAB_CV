# -*- coding: utf-8 -*-
"""
该模块:`dataset_io`包含读取数据集以及数据集分割的类和函数。
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0

from itertools import product
from pickle import load

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

def show_history(history, title="Learning curves", EPOCHS=None):
    """
    单个模型，单词训练。
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["loss"], "o-",
             label="Train loss (%.2f)" % history["loss"][-1])
    plt.plot(history["val_loss"], "o-",
             label="Valid loss (%.2f)" % history["val_loss"][-1])
    if EPOCHS is not None:
        plt.vlines(EPOCHS, 0, history["loss"][0],colors = "c", linestyles = "dashed",label="steps gap")
    plt.legend(loc="best", shadow=1)
    plt.title(title+" loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
#    plt.ylim((-0.05, 5.))
    plt.savefig(title+" loss.jpg")

    plt.figure()
    plt.plot(history["acc"], "o-",
             label="Train accuracy (%.2f%%)" % (history["acc"][-1]*100))
    plt.plot(history["val_acc"], "o-",
             label="Valid accuracy (%.2f%%)" % (history["val_acc"][-1]*100))
    if EPOCHS is not None:
        plt.vlines(EPOCHS, 0, history["acc"][-1],colors = "c", linestyles = "dashed",label="steps gap")
    plt.legend(loc="best", shadow=1)
    plt.title(title+" accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.ylim((-0.05, 1.05))
    plt.savefig(title+" accuracy.jpg")

    return plt


def show_cross_history(path, title="Learning curves"):
    """
    单个模型，交叉验证。
    """
    accs = []
    val_accs = []
    for i in range(3):
        f = open(path+"20180711_validSplit."+str(i)+"_result.h5", "rb")
        contact = load(f)
        accs.append(contact["history"]["acc"])
        val_accs.append(contact["history"]["val_acc"])
        f.close()

    plt.figure()
    len_acc = np.max([len(acc) for acc in accs])
    for i in range(len(accs)):
        accs[i] = accs[i]+(len_acc-len(accs[i]))*[accs[i][-1]]
        val_accs[i] += (len_acc-len(val_accs[i]))*[val_accs[i][-1]]
        plt.plot(val_accs[i], lw=1, alpha=0.3,
                 label="Accuracy for %s fold (%.2f%%)" % (i, val_accs[i][-1]*100))

    accs_mean = np.mean(accs, axis=0)
    accs_std = np.std(accs, axis=0)
    val_accs_mean = np.mean(val_accs, axis=0)
    val_accs_std = np.std(val_accs, axis=0)

    plt.fill_between(range(len_acc),
                     accs_mean - accs_std,
                     accs_mean + accs_std,
                     alpha=0.2, color="r")
    plt.fill_between(range(len_acc),
                     val_accs_mean - val_accs_std,
                     val_accs_mean + val_accs_std,
                     alpha=0.2, color="g",
                     label=r"$\pm$ 1 std. dev.")
    plt.plot(accs_mean, "o-", alpha=0.8, color="r",
             label=r"Training score (%.2f $\pm$ %.2f%%)" % (
                 accs_mean[-1]*100, accs_std[-1]*100))
    plt.plot(val_accs_mean, "o-", alpha=0.8, color="g",
             label=r"Cross-validation score (%.2f $\pm$ %.2f%%)" % (
                 val_accs_mean[-1]*100, val_accs_std[-1]*100))

    plt.grid()
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="best", fontsize=10, shadow=1)
    plt.savefig(title+".jpg")

    return plt


def show_cross_historys(paths, title="Learning curves", subtitles=None):
    """
    多个模型，交叉验证。
    """
    if subtitles is None:
        subtitles = []
        for i in range(len(paths)):
            subtitles.append("Curves"+str(i))

    val_accss = []
    for path in paths:
        val_accs = []
        for i in range(3):
            f = open(path+"20180710_validSplit."+str(i)+"_result.h5", "rb")
            contact = load(f)
            val_accs.append(contact["history"]["val_acc"])
            f.close()
        val_accss.append(val_accs)

    plt.figure()
    len_acc = np.max([len(val_acc)
                      for val_acc in val_accs for val_accs in val_accss])

    colors = ["g", "r", "b", "c", "m", "y", "k", "w"]
    for i in range(len(paths)):
        for j in range(len(val_accss[i])):
            val_accss[i][j] += (len_acc-len(val_accss[i][j])
                                )*[val_accss[i][j][-1]]

        val_accs_mean = np.mean(val_accss[i], axis=0)
        val_accs_std = np.std(val_accss[i], axis=0)

        plt.fill_between(range(len_acc),
                         val_accs_mean - val_accs_std,
                         val_accs_mean + val_accs_std,
                         alpha=0.2, color=colors[i],)
        plt.plot(val_accs_mean, "o-", alpha=0.8, color=colors[i],
                 label=subtitles[i]+r" (%.2f $\pm$ %.2f%%)" % (
            val_accs_mean[-1]*100, val_accs_std[-1]*100))

    plt.grid()
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="best", fontsize=14, shadow=1)

    plt.savefig(title+".jpg")

    return plt


def worst_samples(imgs_valid, labels_valid, score_predict, class_to_index, top=10, names_valid=None):
    """
    dsd
    """
    index_to_class = reverse_dict(class_to_index)
    labels_valid_onehot = to_categorical(labels_valid, len(class_to_index))
    score_valid = np.max(np.array(score_predict)*labels_valid_onehot, axis=1)
    labels_predict = np.argmax(score_predict, axis=1)
    score_predict_max = np.max(score_predict, axis=1)
    worst_index = np.argsort(-score_predict_max+score_valid)[:top]
    asd=score_predict_max-score_valid

    imgs = []
    suptitles = []
    for _, index in enumerate(worst_index):
        imgs.append(imgs_valid[index])
        suptitle = "Predict: %s (%5.2f%%)\n True  : %s (%5.2f%%)" % (
                index_to_class[labels_predict[index]],
                score_predict_max[index]*100,
                index_to_class[labels_valid[index]],
                score_valid[index]*100)
        suptitles.append(suptitle)
        if names_valid is not None:
            name = "Predict %s True %s" % (
                    index_to_class[labels_predict[index]],
                    index_to_class[labels_valid[index]])
            print("%5.2f"%(asd[index]*100))
            print(name+"\n"+names_valid[index])
            print()
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


def classification_report(labels_valid, labels_predict, class_to_index):
    """
    dsd
    """
    target_names = np.array(list(class_to_index.keys()))
    report=metrics.classification_report(labels_valid, labels_predict,
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


def confusion_matrix(labels_valid, labels_predict, class_to_index):
    """
    dsd
    """
    cnf_matrix = metrics.confusion_matrix(labels_valid, labels_predict)

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


def violinBox(labels_valid, labels_predict, class_to_index,section=None):
    plt.style.use("ggplot")
    plt.figure(figsize=(13, 9))

    all_data=[]
    if section is not None:
        for i in range(section[0],section[1]):
            all_data.append(labels_predict[labels_valid==i]+1)
    else:
        for i in range(len(class_to_index)):
            all_data.append(labels_predict[labels_valid==i]+1)

    plt.violinplot(all_data,
                       showmeans=True,
                       showmedians=True)
    plt.boxplot(all_data)

    plt.title('violin-box plot')
    plt.xlabel('True')
    plt.ylabel('Predict')

    if section is not None:
        plt.xticks([y+1 for y in range(len(all_data))],list(class_to_index.keys())[section[0]:section[1]])
        plt.yticks([y+1 for y in range(len(all_data))],list(class_to_index.keys())[section[0]:section[1]])
    else:
        plt.xticks([y+1 for y in range(len(all_data))],list(class_to_index.keys()))
        plt.yticks([y+1 for y in range(len(all_data))],list(class_to_index.keys()))

    plt.savefig("violin-box plot.png")


def violinBoxCompare(labels_valids, labels_predicts, class_to_index,subtitles,section=None):
    colors=['r', 'b', 'g', 'y']

    long = len(class_to_index)
    if section is not None:
        long=section[1]-section[0]

    positionss=[]
    for i in range(len(labels_valids)):
        positionss.append(range(i,long*len(labels_valids),len(labels_valids)))

    plt.style.use("ggplot")
    plt.figure(figsize=(13, 9))

    for j,labels_valid in enumerate(labels_valids):

        labels_predict = labels_predicts[j]
        positions = positionss[j]
        subtitle = subtitles[j]

        all_data=[]
        if section is not None:
            for i in range(section[0],section[1]):
                all_data.append(labels_predict[labels_valid==i]+1-section[0])
        else:
            for i in range(len(class_to_index)):
                all_data.append(labels_predict[labels_valid==i]+1-section[0])

        plt.violinplot(all_data,positions=positions,
                           showmeans=True,
                           showmedians=True)
        plt.boxplot(all_data,positions=positions,)
        plt.plot([],c=colors[j],alpha=.5,lw=3,label=subtitle)

    plt.legend(loc="best", fontsize=14, shadow=1)
    plt.title('violin-box plot')
    plt.xlabel('True')
    plt.xlim([-0.5, long*len(labels_valids)-0.5])
    plt.ylabel('Predict')

    if section is not None:
        plt.xticks([len(labels_valids)*y+0.5*(len(labels_valids)-1) for y in range(len(all_data))],list(class_to_index.keys())[section[0]:section[1]])
        plt.yticks([y+1 for y in range(len(all_data))],list(class_to_index.keys())[section[0]:section[1]])
    else:
        plt.xticks([len(labels_valids)*y+0.5*(len(labels_valids)-1) for y in range(len(all_data))],list(class_to_index.keys()))
        plt.yticks([y+1 for y in range(len(all_data))],list(class_to_index.keys()))

    plt.savefig("violin-box plot.png")



def ROC(labels_valid, score_predict, class_to_index, section=None):
    """
    dsd
    """
    labels_predict = np.argmax(score_predict, axis=1)

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
            tmp = labels_predict == i
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

    if section is not None:
        colors = ["c", "g", "r", "m", "k"] # c y
        for i in range(1, len(section)):
            mean_tpr = np.mean(tprs[section[i-1]:section[i]], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs[section[i-1]:section[i]], axis=0)
            plt.plot(mean_fpr, mean_tpr, color=colors[i],
                     label=r"Mean ROC for %s (mAUC = %.2f $\pm$ %.2f)" % (
                             list(class_to_index.keys())[section[i-1]][0],
                             mean_auc, std_auc),
                     lw=2, alpha=1)

    #mean_tpr = np.average(tprs, weights=vldPerClass, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    #std_auc = weighted_std(aucs, vldPerClass)
    std_auc = np.std(aucs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color="c",
             label=r"Mean ROC (mAUC = %.2f $\pm$ %.2f)" % (mean_auc, std_auc),
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

def ROCCompare(labels_valids, scores_predicts, class_to_index, subtitles, section=None):
    """
    dsd
    """
    plt.style.use("ggplot")

    plt.figure(figsize=(13, 9))
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r",
             label="Luck", alpha=.8)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    start=0
    end=len(class_to_index)
    if section is not None:
        start=section[0]
        end=section[1]


    for i,labels_valid in enumerate(labels_valids):

        scores_predict = scores_predicts[i]
        subtitle = subtitles[i]

        indexs_sample = labels_valid == start
        for k in range(start, end):
            tmp = labels_valid == k
            indexs_sample = indexs_sample | tmp

        for i in range(start,end):
            fpr, tpr, thresholds = metrics.roc_curve(
                labels_valid[indexs_sample],
                scores_predict[indexs_sample][:, i],
                pos_label=i)
            fpr[np.isnan(fpr)] = 0.
            fpr[-1] = 1.
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)

        #mean_tpr = np.average(tprs, weights=vldPerClass, axis=0)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        #std_auc = weighted_std(aucs, vldPerClass)
        std_auc = np.std(aucs, axis=0)
        plt.plot(mean_fpr, mean_tpr,
                 label=r"Mean ROC for %s (mAUC = %.2f $\pm$ %.2f)" % (subtitle, mean_auc, std_auc),
                 lw=3, alpha=1)

        #std_tpr = weighted_std(tprs, vldPerClass, axis=0)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                         alpha=.2)

    plt.legend(loc="best", fontsize=14, shadow=1)
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
    labels_predict = np.argmax(score_predict, axis=1)

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
            tmp = labels_predict == i
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
                 label="PR for %s (AP = %.2f)" % (list(class_to_index.keys())[i], metrics.auc(R, P)))

    if section is not None:
        colors = ["c", "g", "r", "m", "k"] # c y
        for i in range(1, len(section)):
            mean_precision = np.mean(precisions[section[i-1]:section[i]], axis=0)
            mean_precision_recall_auc = metrics.auc(mean_recall, mean_precision)
            std_precision_recall_auc = np.std(precision_recall_auc[section[i-1]:section[i]], axis=0)
            plt.plot(mean_recall, mean_precision, color=colors[i],
                     label=r"Mean ROC for %s (mAP = %.2f $\pm$ %.2f)" % (
                             list(class_to_index.keys())[section[i-1]][0],
                             mean_precision_recall_auc, std_precision_recall_auc),
                     lw=2, alpha=1)

    #mean_precision = np.average(precisions, weights=vldPerClass, axis=0)
    mean_precision = np.mean(precisions, axis=0)
    mean_precision_recall_auc = metrics.auc(mean_recall, mean_precision)
    #std_precision_recall_auc = weighted_std(precision_recall_auc, vldPerClass)
    std_precision_recall_auc = np.std(precision_recall_auc)
    plt.plot(mean_recall, mean_precision, color="c", lw=2, alpha=.8,
             label=r"Mean PR (mAP = %.2f $\pm$ %.2f)" % (
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


def PRCompare(labels_valids, scores_predicts, class_to_index, subtitles, section=None):
    """
    dsd
    """
    plt.style.use("ggplot")

    plt.figure(figsize=(13, 9))
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

    start=0
    end=len(class_to_index)
    if section is not None:
        start=section[0]
        end=section[1]


    for i,labels_valid in enumerate(labels_valids):

        scores_predict = scores_predicts[i]
        subtitle = subtitles[i]

        indexs_sample = labels_valid == start
        for k in range(start, end):
            tmp = labels_valid == k
            indexs_sample = indexs_sample | tmp

        for i in range(start,end):
            P, R, _ = metrics.precision_recall_curve(
                labels_valid[indexs_sample],
                scores_predict[indexs_sample][:, i],
                pos_label=i)
            precision.append(P)
            recall.append(R)
            precision_recall_auc.append(metrics.auc(R, P))
            precisions.append(
                interp(mean_recall, np.flip(R, axis=0), np.flip(P, axis=0)))

        #mean_precision = np.average(precisions, weights=vldPerClass, axis=0)
        mean_precision = np.mean(precisions, axis=0)
        mean_precision_recall_auc = metrics.auc(mean_recall, mean_precision)
        #std_precision_recall_auc = weighted_std(precision_recall_auc, vldPerClass)
        std_precision_recall_auc = np.std(precision_recall_auc)
        plt.plot(mean_recall, mean_precision, lw=3, alpha=1,
                 label=r"Mean PR for %s (mAP = %.2f $\pm$ %.2f)" % (
                         subtitle,
                         mean_precision_recall_auc, std_precision_recall_auc))

        #std_precisions = weighted_std(precisions, vldPerClass, axis=0)
        std_precisions = np.std(precisions, axis=0)
        precision_upper = np.minimum(mean_precision + std_precisions, 1)
        precision_lower = np.maximum(mean_precision - std_precisions, 0)
        plt.fill_between(mean_recall, precision_lower, precision_upper,
                          alpha=.2)

    plt.legend(loc="best", fontsize=14, shadow=1)
    plt.title("Precision-Recall curve")
    plt.xlabel("Recall")
    plt.xlim([-.05, 1.05])
    plt.ylabel("Precision")
    plt.ylim([-.05, 1.05])
    plt.savefig("Precision-Recall curve")

