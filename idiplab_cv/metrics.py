# -*- coding: utf-8 -*-
'''
该模块:`dataset_io`包含读取数据集以及数据集分割的类和函数。

'''

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0

#    ['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', '_classic_test']


from itertools import product
from pickle import load
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import metrics

from idiplab_cv.dataset_io import reverse_dict
from idiplab_cv.dataset_io import to_categorical

if not os.path.exists("images"):
    os.mkdir("images")

# %%


def _weighted_std(values, weights, axis=None):
    average = np.average(values, weights=weights, axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return np.sqrt(variance)


def _special_average_precision_auc(precision, recall, axis=None):
    '''
    计算AP时的特殊处理。一般PR，AUC都可以用:
        A, B, thresholds = A_B_curve()
        aera = auc(A,B)
    但sklearn在计算precision时会将最后一个值设为1，所以需要特殊处理。
    '''
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def _plot_confusion_matrix(cm, class_to_index,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
    '''
    dsd
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_to_index))
    plt.xticks(tick_marks, class_to_index, rotation=45)
    plt.yticks(tick_marks, class_to_index)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predict')
    return plt

# %%


def decode_predictions(preds, class_index, top=5):
    '''
    dsd
    '''
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(str(i), class_index[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def show_history_pair(history, title="Learning curves", EPOCHS=None):
    """
    画出单个模型，单次训练下的训练曲线。
    需要输入history数据。
    用2个图，一个图表示训练测试的损失，另一个表示训练测试的准确率。
    如果给出EPOCHS，则画出分界线
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["loss"], "o-",
             label="Train loss (%.2f)" % history["loss"][-1])
    plt.plot(history["val_loss"], "o-",
             label="Valid loss (%.2f)" % history["val_loss"][-1])
    if EPOCHS is not None:
        plt.vlines(EPOCHS, 0, history["loss"][0], colors="c",
                   linestyles="dashed", label="steps gap")
    plt.legend(loc="best", shadow=1)
    plt.title(title+" loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
#    plt.ylim((-0.05, 5.))
    plt.savefig("images/"+title+" loss.jpg")

    plt.figure()
    plt.plot(history["acc"], "o-",
             label="Train accuracy (%.2f%%)" % (history["acc"][-1]*100))
    plt.plot(history["val_acc"], "o-",
             label="Valid accuracy (%.2f%%)" % (history["val_acc"][-1]*100))
    if EPOCHS is not None:
        plt.vlines(EPOCHS, 0, history["acc"][-1], colors="c",
                   linestyles="dashed", label="steps gap")
    plt.legend(loc="best", shadow=1)
    plt.title(title+" accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.ylim((-0.05, 1.05))
    plt.savefig("images/"+title+" accuracy.jpg")

    return plt


def show_history_section(history, EPOCHS, test_acc=None, title='Learning curves'):
    '''
    画出单个模型，单次训练下的训练曲线。
    需要输入history数据，EPOCHS。
    用1个图，表示测试集的损失和准确率，画出分界线
    如果给出test_acc，则画出每个阶段时测试集的准确率
    '''
    fig = plt.figure(figsize=(12, 9))
    plt.style.use('default')

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
#    ax1.plot(history['loss'], 'o-',
#             label='train_loss (%.4f)' % (np.min(history['loss'])))
    ax1.plot(history['val_loss'], 'o-', c=plt.cm.Set1(1),
             label='val_loss (%.4f)' % (np.min(history['val_loss'])))
    ax2.plot(np.nan, 'o-', c=plt.cm.Set1(1),
             label='val_loss (%.4f)' % (np.min(history['val_loss'])))
#    ax2.plot(history['acc'], 'o-',
#             label='train_acc (%.2f%%)' % (np.max(history['acc'])*100))
    ax2.plot(history['val_acc'], 'o-', c=plt.cm.Set1(0),
             label='val_acc (%.2f%%)' % (np.max(history['val_acc'])*100))

    if test_acc is not None:

        size = 14
        color = plt.cm.Set1(0.25)
        linestyles = 'dashed'
        bbox = dict(boxstyle='round', alpha=0.75, fc=color)

        for i in range(len(test_acc)):
            if i == 0:
                low = 0
            else:
                low = EPOCHS[i-1]

            ax2.text(EPOCHS[i]-30, test_acc[i]+0.03, 'test_acc:%.2f%%' %
                     (test_acc[i]*100), ha='left', size=size, bbox=bbox)
            ax2.hlines(test_acc[i], low, EPOCHS[i], lw=3,
                       color=color, linestyles=linestyles, zorder=3)

        plt.hlines(np.nan, np.nan, np.nan, lw=3,
                   color=color, linestyles=linestyles, label='test_acc', zorder=3)
        plt.vlines(EPOCHS[:-1], 0, history['loss'][0],  lw=3, colors=plt.cm.Set1(0.75),
                   linestyles='dashed', label='steps gap')

    ax1.grid(True)
#    ax1.grid(True)
    plt.title(title)
    ax1.set_xlim(-0, EPOCHS[-1])
    ax1.set_ylim(-0.0, 2.)
    ax2.set_ylim(-0.0, 1.0)
    ax2.spines['right'].set_color(plt.cm.Set1(0))
    ax2.tick_params(axis='y', colors=plt.cm.Set1(0), labelsize=13)
    ax2.spines['left'].set_color(plt.cm.Set1(1))
    ax1.tick_params(axis='x', labelsize=13)
    ax1.tick_params(axis='y', colors=plt.cm.Set1(1), labelsize=13)
    ax1.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('Accuracy', color=plt.cm.Set1(0), fontsize=15)
    ax1.set_ylabel('Categorical Crossentropy Loss',
                   color=plt.cm.Set1(1), fontsize=15)
    plt.legend(loc='lower right', fontsize=14, shadow=1)
    plt.savefig("images/"+title+'.png', dpi=200, bbox_inches='tight')

    return fig

#    ['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', '_classic_test']


def show_history_cross(historys, test_accs=None, title='Learning curves', EPOCHS=None):
    '''
    画出单个模型，多次训练下的训练曲线。
    需要输入historys数据，EPOCHS。
    用1个图，表示测试集的损失和准确率以及波动范围，画出分界线
    如果给出test_acc，则画出每个阶段时测试集的准确率
    '''
    val_loss = []
    val_accs = []
    for history in historys:
        val_loss.append(history['val_loss'])
        val_accs.append(history['val_acc'])

    fig = plt.figure(figsize=(12, 9))
    plt.style.use('default')
    len_acc = np.max([len(val_acc) for val_acc in val_accs])
#    for i in range(len(accs)):
#        accs[i] = accs[i]+(len_acc-len(accs[i]))*[accs[i][-1]]
#        val_accs[i] += (len_acc-len(val_accs[i]))*[val_accs[i][-1]]
#        plt.plot(val_accs[i], lw=1, alpha=0.3,
#                 label='Accuracy for %s fold (%.2f%%)' % (i, val_accs[i][-1]*100))

    val_loss_mean = np.mean(val_loss, axis=0)
    val_loss_std = np.std(val_loss, axis=0)
    val_accs_mean = np.mean(val_accs, axis=0)
    val_accs_std = np.std(val_accs, axis=0)

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.fill_between(range(len_acc),
                     val_loss_mean - val_loss_std,
                     val_loss_mean + val_loss_std,
                     alpha=0.2, color=plt.cm.Set1(1))
    ax2.fill_between(range(len_acc),
                     val_accs_mean - val_accs_std,
                     val_accs_mean + val_accs_std,
                     alpha=0.2, color=plt.cm.Set1(0),)
#                     label=r'$\pm$ 1 std. dev.')
    space = np.linspace(0, len_acc-1, num=80, endpoint=True, dtype='int')
    ax1.plot(space, val_loss_mean[space], 'o-',
             alpha=0.5, color=plt.cm.Set1(1))
    ax2.plot(np.nan, 'o-', alpha=0.5, color=plt.cm.Set1(1),
             label='Validation categorical cross-entropy loss: %.2f%%' % (val_loss_mean[-1]*100))
    ax2.plot(space, val_accs_mean[space], 'o-', alpha=0.5, color=plt.cm.Set1(0),
             label='Validation accuracy: %.2f%%' % (val_accs_mean[-1]*100))

    if test_accs is not None:
        test_acc_mean = np.mean(test_accs, axis=0)
        test_acc_std = np.std(test_accs, axis=0)

        size = 14
        color = plt.cm.Set1(0.25)
        linestyles = 'dashed'
        bbox = dict(boxstyle='round', alpha=0.75, fc=color)

        for i in range(len(test_accs)+1):
            if i == 0:
                low = 0
            else:
                low = EPOCHS[i-1]

            ax2.fill_between([low, EPOCHS[i]],
                             [test_acc_mean[i] - test_acc_std[i],
                                 test_acc_mean[i] - test_acc_std[i]],
                             [test_acc_mean[i] + test_acc_std[i],
                                 test_acc_mean[i] + test_acc_std[i]],
                             color=color, alpha=0.5, zorder=2)
            ax2.text(EPOCHS[i]-15, test_acc_mean[i]+0.05, '%.2f%%' %
                     (test_acc_mean[i]*100), ha='left', size=size, bbox=bbox)
            ax2.hlines(test_acc_mean[i], low, EPOCHS[i], lw=3,
                       color=color, linestyles=linestyles, zorder=3)

        plt.hlines(np.nan, np.nan, np.nan, lw=3,
                   color=color, linestyles=linestyles, label='Test accuracy', zorder=3)
        plt.vlines(EPOCHS[:-1], 0, 1, lw=3, colors=plt.cm.Set1(0.75),
                   linestyles='dashed', label='Stage interval')

    ax1.grid(True)
#    ax1.grid(True)
#    plt.title(title)
    ax1.set_xlim(-0, EPOCHS[-1])
    ax1.set_ylim(-0.0, 2.)
    ax2.set_ylim(-0.0, 1.0)
    ax2.spines['right'].set_color(plt.cm.Set1(0))
    ax2.tick_params(axis='y', colors=plt.cm.Set1(0), labelsize=13)
    ax2.spines['left'].set_color(plt.cm.Set1(1))
    ax1.tick_params(axis='x', labelsize=13)
    ax1.tick_params(axis='y', colors=plt.cm.Set1(1), labelsize=13)
    ax1.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('Accuracy', color=plt.cm.Set1(0), fontsize=15)
    ax1.set_ylabel('Categorical Cross-Entropy Loss',
                   color=plt.cm.Set1(1), fontsize=15)
    plt.legend(loc='lower right', fontsize=14, shadow=1)
    plt.savefig("images/"+title+'.png', dpi=200, bbox_inches='tight')

    return fig


def show_historys_compare(historyss, title='Learning curves', subtitles=None):
    '''
    画出多个模型，每个模型多次训练下的训练曲线。
    需要输入result.h5地址，函数自己读取训练记录。
    用1个图，表示准确率的平均值和一个方差的波动范围
    '''
    if subtitles is None:
        subtitles = []
        for i in range(len(historyss)):
            subtitles.append('Curves'+str(i))

    val_accss = []
    for historys in historyss:
        val_accs = []
        for history in historys:
            val_accs.append(history['val_acc'])
        val_accss.append(val_accs)

    plt.figure()
    len_acc = np.max([len(val_acc)
                      for val_acc in val_accs for val_accs in val_accss])

    colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k', 'w']
    for i in range(len(historyss)):
        for j in range(len(val_accss[i])):
            val_accss[i][j] += (len_acc-len(val_accss[i][j])
                                )*[val_accss[i][j][-1]]

        val_accs_mean = np.mean(val_accss[i], axis=0)
        val_accs_std = np.std(val_accss[i], axis=0)

        plt.fill_between(range(len_acc),
                         val_accs_mean - val_accs_std,
                         val_accs_mean + val_accs_std,
                         alpha=0.2, color=colors[i],)
        plt.plot(val_accs_mean, 'o-', alpha=0.8, color=colors[i],
                 label=subtitles[i]+r' (%.2f $\pm$ %.2f%%)' % (
            val_accs_mean[-1]*100, val_accs_std[-1]*100))

    plt.grid()
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.legend(loc='best', fontsize=14, shadow=1)

    plt.savefig("images/"+title+'.png', bbox_inches='tight', dpi=200)

    return plt


def number_per_class(labels_valid, class_to_index):
    '''
    dsd
    '''
    vldPerClass = [np.sum(labels_valid == i) for i in range(len(labels_valid))]
    plt.figure()
    plt.bar(range(len(vldPerClass)), vldPerClass,
            color='rgb', tick_label=np.array(list(class_to_index.keys())))
    plt.title('Number per class')
    plt.savefig("images/"+'Number for class all.png',
                bbox_inches='tight', dpi=200)
    return plt


def classification_report(labels_valid, labels_predict, class_to_index, title="classification_report"):
    '''
    基本分类指标
    '''
    target_names = np.array(list(class_to_index.keys()))
    report = metrics.classification_report(labels_valid, labels_predict,
                                           digits=4, target_names=target_names)
    print(report)
    f = open("images/"+title+'.txt', 'w')
    f.write(report)
    f.close()
    report_dict = metrics.classification_report(labels_valid, labels_predict,
                                                digits=4, target_names=target_names, output_dict=True)

    return report_dict


def confusion_matrix(labels_valid, labels_predict, class_to_index):
    '''
    dsd
    '''
    cnf_matrix = metrics.confusion_matrix(labels_valid, labels_predict)

    plt.figure(figsize=(13, 7))
    plt.style.use('default')

#    ['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', '_classic_test']
    plt.subplot(121)
    _plot_confusion_matrix(
        cnf_matrix,
        class_to_index=np.array(list(class_to_index.keys())),
        title='Plain confusion matrix')
    plt.subplot(122)
    _plot_confusion_matrix(
        cnf_matrix,
        class_to_index=np.array(list(class_to_index.keys())),
        normalize=True,
        title='Normalized confusion matrix')

    plt.savefig("images/"+'Confusion matrix.png', bbox_inches='tight', dpi=200)
    return plt, cnf_matrix


def violinBox(labels_valid, labels_predict, class_to_index, section=None):
    '''箱型图

        XXXX

    Args:
        labels_valid (:obj:`array` of :obj:`int`): 正确标签数据。请输入需要一维向量，例如np.array([0, 0, 1, 1])。
        labels_predict (:obj:`array` of :obj:`int`): 预测标签数据。请输入需要一维向量，例如np.array([0, 0, 1, 1])。
        class_to_index (:obj:`dict` of :obj:`str` to :obj:`int`): 各类对应的标签序号

    Examples:

        >>> labels_valid = np.array([0, 0, 1, 1])
        >>> labels_predict = np.array([0, 0, 0, 1])
        >>> class_to_index = {'A':0, 'B':1}
        >>> violinBox(labels_valid, labels_predict, class_to_index)

    '''
    plt.style.use('ggplot')
    plt.figure(figsize=(13, 9))

    all_data = []
    if section is not None:
        for i in range(section[0], section[1]):
            all_data.append(labels_predict[labels_valid == i]+1)
    else:
        for i in range(len(class_to_index)):
            all_data.append(labels_predict[labels_valid == i]+1)

    plt.violinplot(all_data,
                   showmeans=True,)
#                   showmedians=True)
    plt.boxplot(all_data,
                meanline=True)


#    plt.title('violin plot', fontsize=25)
    plt.xlabel('True', fontsize=20)
    plt.ylabel('Predict', fontsize=20)

    if section is not None:
        plt.xticks([y+1 for y in range(len(all_data))],
                   list(class_to_index.keys())[section[0]:section[1]])
        plt.yticks([y+1 for y in range(len(all_data))],
                   list(class_to_index.keys())[section[0]:section[1]])
    else:
        plt.xticks([y+1 for y in range(len(all_data))],
                   list(class_to_index.keys()))
        plt.yticks([y+1 for y in range(len(all_data))],
                   list(class_to_index.keys()))

    plt.xticks(fontsize=15)
    plt.xlim(0.5, len(class_to_index)+0.5)
    plt.yticks(fontsize=15)
    plt.ylim(0.5, len(class_to_index)+0.5)

    plt.savefig("images/"+'violin-box plot.png', bbox_inches='tight', dpi=200)


def ROC(labels_valid, score_predict, class_to_index, section=None):
    '''
    dsd
    '''
    labels_predict = np.argmax(score_predict, axis=1)

    plt.figure(figsize=(15, 9))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    tprs = []
    ROC_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(class_to_index)):

        # ddddddddddddddddddd
        if section is None:
            #            indexs_sample = labels_valid == i
            #            tmp = labels_predict == i
            #            indexs_sample = indexs_sample | tmp
            indexs_sample = range(len(labels_valid))
        else:
            #            indexs_sample = labels_valid == i
            #            tmp = labels_predict == i
            #            indexs_sample = indexs_sample | tmp
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
        if len(tpr) > 1:
            ROC_aucs.append(metrics.auc(fpr, tpr))
            tprs.append(interp(mean_fpr, fpr, tpr))
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC for %s (AUC = %.4f)' % (list(class_to_index.keys())[i], metrics.auc(fpr, tpr)))

    if section is not None:
        colors = ['c', 'g', 'r', 'm', 'k']  # c y
        for i in range(1, len(section)):
            mean_tpr = np.mean(tprs[section[i-1]:section[i]], axis=0)
#            mean_tpr[-1] = 1.0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(ROC_aucs[section[i-1]:section[i]], axis=0)
            plt.plot(mean_fpr, mean_tpr, color=colors[i],
                     label=r'Mean ROC for %s (mAUC = %.2f $\pm$ %.2f)' % (
                list(class_to_index.keys())[section[i-1]][0],
                mean_auc, std_auc),
                lw=2, alpha=1)

    vldPerClass = [np.sum(labels_valid == i)
                   for i in range(len(class_to_index))]
    mean_tpr = np.average(tprs, weights=vldPerClass, axis=0)
#    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = _weighted_std(ROC_aucs, vldPerClass)
#    std_auc = np.std(ROC_aucs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='c',
             label=r'Mean ROC (mAUC = %.4f $\pm$ %.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = _weighted_std(tprs, vldPerClass, axis=0)
#    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                     color='c', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    fig = plt.gcf()
    fig.subplots_adjust(right=.65)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
               fontsize=14, shadow=1)
    plt.title('Receiver operating characteristic curve')
    plt.xlabel('False Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.ylim([-0.05, 1.05])
    plt.savefig("images/"+'Receiver operating characteristic curve.png',
                bbox_inches='tight', dpi=200)
    return mean_auc


def ROC_compare(labels_valids, scores_predicts, class_to_index, subtitles, section=None):
    '''
    dsd
    '''
    plt.style.use('ggplot')

    plt.figure(figsize=(13, 9))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    tprs = []
    ROC_aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    start = 0
    end = len(class_to_index)
    if section is not None:
        start = section[0]
        end = section[1]

    for i, labels_valid in enumerate(labels_valids):

        scores_predict = scores_predicts[i]
        subtitle = subtitles[i]

        indexs_sample = labels_valid == start
        for k in range(start, end):
            tmp = labels_valid == k
            indexs_sample = indexs_sample | tmp

        for i in range(start, end):
            fpr, tpr, thresholds = metrics.roc_curve(
                labels_valid[indexs_sample],
                scores_predict[indexs_sample][:, i],
                pos_label=i)
            ROC_aucs.append(metrics.auc(fpr, tpr))
            tprs.append(interp(mean_fpr, fpr, tpr))

        #mean_tpr = np.average(tprs, weights=vldPerClass, axis=0)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        #std_auc = weighted_std(aucs, vldPerClass)
        std_auc = np.std(ROC_aucs, axis=0)
        plt.plot(mean_fpr, mean_tpr,
                 label=r'Mean ROC for %s (mAUC = %.2f $\pm$ %.2f)' % (
                     subtitle, mean_auc, std_auc),
                 lw=3, alpha=1)

        #std_tpr = weighted_std(tprs, vldPerClass, axis=0)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                         alpha=.2)

    plt.legend(loc='best', fontsize=14, shadow=1)
    plt.title('Receiver operating characteristic curve')
    plt.xlabel('False Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.ylim([-0.05, 1.05])
    plt.savefig("images/"+'Receiver operating characteristic curve.png',
                bbox_inches='tight', dpi=200)


def PR(labels_valid, score_predict, class_to_index, section=None, title="Precision-Recall curve"):
    '''
    dsd
    '''
    labels_predict = np.argmax(score_predict, axis=1)

    plt.figure(figsize=(15, 9))
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1, 10000)
        y = f_score * x / (2 * x - f_score)
        x = x[y <= 1]
        y = y[y <= 1]
        if f_score == 0.2:
            plt.plot(x[y >= 0], y[y >= 0], color='gray',
                     alpha=0.6, label='Iso-f1 curves')
        else:
            plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.6)
        plt.annotate('f1=%.1f' % f_score, xy=(0.9, y[-1] + .02))

    precisions = []
    precision_recall_aucs = []
    mean_recall = np.linspace(0, 1, 100)

    for i in range(len(class_to_index)):

        # ddddddddddddddddddd
        if section is None:
            indexs_sample = labels_valid == i
            tmp = labels_predict == i
            indexs_sample = indexs_sample | tmp
        else:
            indexs_sample = labels_valid == i
            tmp = labels_predict == i
            indexs_sample = indexs_sample | tmp
#            for j in range(1, len(section)):
#                if i < section[j]:
#                    indexs_sample = labels_valid == section[j-1]
#                    for k in range(section[j-1]+1, section[j]):
#                        tmp = labels_valid == k
#                        indexs_sample = indexs_sample | tmp
#                    break

        P, R, _ = metrics.precision_recall_curve(
            labels_valid[indexs_sample],
            score_predict[indexs_sample][:, i],
            pos_label=i)
        precision_recall_aucs.append(_special_average_precision_auc(P, R))
        precisions.append(
            interp(mean_recall, np.flip(R, axis=0), np.flip(P, axis=0)))
        plt.plot(R, P, lw=1, alpha=.3,
                 label='PR for %s (AP = %.4f)' % (list(class_to_index.keys())[i], _special_average_precision_auc(P, R)))

    if section is not None:
        colors = ['c', 'g', 'r', 'm', 'k']  # c y
        for i in range(1, len(section)):
            mean_precision = np.mean(
                precisions[section[i-1]:section[i]], axis=0)
            mean_precision_recall_auc = _special_average_precision_auc(
                mean_precision, -mean_recall)
            std_precision_recall_auc = np.std(
                precision_recall_aucs[section[i-1]:section[i]], axis=0)
            plt.plot(mean_recall, mean_precision, color=colors[i],
                     label=r'Mean ROC for %s (mAP = %.4f $\pm$ %.4f)' % (
                list(class_to_index.keys())[section[i-1]][0],
                mean_precision_recall_auc, std_precision_recall_auc),
                lw=2, alpha=1)

    vldPerClass = [np.sum(labels_valid == i)
                   for i in range(len(class_to_index))]
    mean_precision = np.average(precisions, weights=vldPerClass, axis=0)
#    mean_precision = np.mean(precisions, axis=0)
    mean_precision_recall_auc = _special_average_precision_auc(
        mean_precision, -mean_recall)
    std_precision_recall_auc = _weighted_std(
        precision_recall_aucs, vldPerClass)
#    std_precision_recall_auc = np.std(precision_recall_aucs)
    plt.plot(mean_recall, mean_precision, color='c', lw=2, alpha=.8,
             label=r'Mean PR (mAP = %.4f $\pm$ %.4f)' % (
                 mean_precision_recall_auc, std_precision_recall_auc))

    std_precisions = _weighted_std(precisions, vldPerClass, axis=0)
#    std_precisions = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precisions, 1)
    precision_lower = np.maximum(mean_precision - std_precisions, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper,
                     color='c', alpha=.2, label=r'$\pm$ 1 std. dev.')

    fig = plt.gcf()
    fig.subplots_adjust(right=.65)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
               fontsize=14, shadow=1)
    plt.title('Precision-Recall curve')
    plt.xlabel('Recall')
    plt.xlim([-.0, 1.0])
    plt.ylabel('Precision')
    plt.ylim([-.0, 1.0])
    plt.savefig("images/"+title+'.png', bbox_inches='tight', dpi=200)
    return mean_precision_recall_auc


def PR_compare(labels_valids, scores_predicts, class_to_index, subtitles, section=None, title="Precision-Recall curve"):
    '''
    dsd
    '''
    plt.style.use('ggplot')

    plt.figure(figsize=(13, 9))
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        if f_score == 0.2:
            plt.plot(x[y >= 0], y[y >= 0], color='gray',
                     alpha=0.6, label='Iso-f1 curves')
        else:
            plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.6)
        plt.annotate('f1=%.1f' % f_score, xy=(0.9, y[45] + .02))

    precisions = []
    precision_recall_aucs = []
    mean_recall = np.linspace(0, 1, 100)

    start = 0
    end = len(class_to_index)
    if section is not None:
        start = section[0]
        end = section[1]

    for i, labels_valid in enumerate(labels_valids):

        scores_predict = scores_predicts[i]
        subtitle = subtitles[i]

        indexs_sample = labels_valid == start
        for k in range(start, end):
            tmp = labels_valid == k
            indexs_sample = indexs_sample | tmp

        for i in range(start, end):
            P, R, _ = metrics.precision_recall_curve(
                labels_valid[indexs_sample],
                scores_predict[indexs_sample][:, i],
                pos_label=i)
            precision_recall_aucs.append(_special_average_precision_auc(P, R))
            precisions.append(
                interp(mean_recall, np.flip(R, axis=0), np.flip(P, axis=0)))

        #mean_precision = np.average(precisions, weights=vldPerClass, axis=0)
        mean_precision = np.mean(precisions, axis=0)
        mean_precision_recall_auc = _special_average_precision_auc(
            mean_precision, mean_recall)
        #std_precision_recall_auc = weighted_std(precision_recall_auc, vldPerClass)
        std_precision_recall_auc = np.std(precision_recall_aucs)
        plt.plot(mean_recall, mean_precision, lw=3, alpha=1,
                 label=r'Mean PR for %s (mAP = %.2f $\pm$ %.2f)' % (
                     subtitle,
                     mean_precision_recall_auc, std_precision_recall_auc))

        #std_precisions = weighted_std(precisions, vldPerClass, axis=0)
        std_precisions = np.std(precisions, axis=0)
        precision_upper = np.minimum(mean_precision + std_precisions, 1)
        precision_lower = np.maximum(mean_precision - std_precisions, 0)
        plt.fill_between(mean_recall, precision_lower, precision_upper,
                         alpha=.2)

    plt.legend(loc='best', fontsize=14, shadow=1)
    plt.title('Precision-Recall curve')
    plt.xlabel('Recall')
    plt.xlim([-.05, 1.05])
    plt.ylabel('Precision')
    plt.ylim([-.05, 1.05])
    plt.savefig("images/"+title+'.png', bbox_inches='tight', dpi=200)


def summary_simple(labels_valid, labels_predict, scores_predict, class_to_index):
    report_dict = classification_report(
        labels_valid, labels_predict, class_to_index)
    plt, cnf_matrix = confusion_matrix(
        labels_valid, labels_predict, class_to_index)
    violinBox(labels_valid, labels_predict, class_to_index, section=None)
    PR(labels_valid, scores_predict, class_to_index)
    ROC(labels_valid, scores_predict, class_to_index)


def summary(labels_valid, labels_predict, scores_predict, class_to_index):
    labels_valid = np.copy(labels_valid)

    report_dict = classification_report(
        labels_valid, labels_predict, class_to_index)
    r0 = report_dict['weighted avg']['recall']
    plt, cnf_matrix = confusion_matrix(
        labels_valid, labels_predict, class_to_index)
    violinBox(labels_valid, labels_predict, class_to_index, section=None)
    auc = ROC(labels_valid, scores_predict, class_to_index)
    mAP = PR(labels_valid, scores_predict, class_to_index)

    r1ss = []
    ess = []
    supports = []
    length = len(cnf_matrix)
    for true, line in enumerate(cnf_matrix):
        support = sum(line)
        low = max(true-1, 0)
        up = min(true+2, length)
        r1 = sum(line[low:up])/support
        mask = abs(np.linspace(0, length-1, length)-true)
        ee = mask@line/support
        r1ss.append(r1)
        ess.append(ee)
        supports.append(support)
    r1 = np.average(r1ss, weights=supports, axis=0)
    ee = np.average(ess, weights=supports, axis=0)

    labels_valid[labels_valid < 2] = 0
    labels_valid[labels_valid >= 2] = 1
    F = np.sum(scores_predict[:, :2], axis=1)
    S = np.sum(scores_predict[:, 2:], axis=1)
    scores_predict = np.array([F, S]).T
    labels_predict = np.argmax(scores_predict, axis=1)
    class_to_index = {'Follow-up': 0, 'Surgery': 1}

    mAPT = PR(labels_valid, scores_predict, class_to_index)
    report_dict = classification_report(
        labels_valid, labels_predict, class_to_index)
    F1 = report_dict['weighted avg']['f1-score']

    print("ee: %.3f" % (ee))
    print("r0: %.3f" % (r0))
    print("re1: %.3f" % (r1))
    print("mAP: %.3f" % (mAP))
    print("F1: %.3f" % (F1))
    print("mAPT: %.3f" % (mAPT))

    return ee, r0, r1, mAP, F1, mAPT
