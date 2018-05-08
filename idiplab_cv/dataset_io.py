# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:02:44 2018

@author: Sandiagal
"""

from PIL import Image
import glob
import os
import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical

# %%


def get_split_index(labels, n_splits, now_splits):
    i = 0
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
        #        print("TRAIN:", train_index, "TEST:", test_index)
        if i == now_splits:
            return train_index, test_index
        i = i+1


def load_all_data(path_orig, incl_agmt=False):
    """
    读取path_orig目录内所有图片，数据集必须按照下面的形式存放
        data/ # path_orig的末端，目录名，自定义
            dogs/ # 类名，自定义
                dog001.jpg # 样本名，自定义
            dogs_agmt/ # dogs类的增强数据，前缀严格对应，自动生成
                dog001_X_XXX.jpg  # dog001样本的增强数据，前缀严格对应，自动生成
            cats/
                cat001.jpg
            cat_agmt/
                cat001_X_XXX.jpg

    Parameters
    ----------
    path_orig : str，数据所在文件夹地址，目录结尾一定要加“/”

    incl_agmt : bool，是否读取增强数据

    Returns
    -------
    imgs_orig : list，（样本数），原始图片集，元素为（宽，高，通道）的图片numpy

    imgs_name_orig : list，（样本数），原始图片名集，元素为样本名str

    labels_orig : list，（样本数），原始图片名集，元素为分类名str

    imgs_agmt : list，（增强样本数），增强图片集，元素为（宽，高，通道）的图片numpy

    imgs_name_agmt : list，（增强样本数），增强图片名集，元素为样本名str

    labels_agmt : list，（增强样本数），增强图片名集，元素为分类名str

    classes : list，（分类数），分类名称，元素为分类名str

    Examples
    --------
      >>> imgs_orig, imgs_name_orig, labels_orig, imgs_agmt, imgs_name_agmt, labels_agmt, classes = load_all_data('../data/', incl_agmt)
    """

    print("--->Start reading")
    start = time.clock()

    imgs_orig = []
    imgs_name_orig = []
    labels_orig = []
    imgs_agmt = []
    imgs_name_agmt = []
    labels_agmt = []
    classes = {}

    g = os.walk(path_orig)

    for path, dir_list, file_list in g:
        # dir_list:类别的list，例如['cats', 'dogs']
        for dir_name in dir_list:
            # dir_name:类别的名称，例如cats

            if dir_name.rfind("_agmt") == -1:
                # 原始数据
                classes[dir_name] = len(classes)

            if dir_name.rfind("_agmt") != -1 and not incl_agmt:
                continue

            for filename in glob.glob(path_orig+dir_name+"/*.jpg"):
                #  filename:带有路径的图片完整地址，例如../Data_Origin/cats\rezero_icon_10.jpg
                img = Image.open(filename)  # 这是一个PIL图像
                img = np.asarray(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)

                if dir_name.rfind("_agmt") == -1:
                    # 原始数据
                    imgs_orig.append(img)  # 把图片数组加到一个列表里面
                    imgs_name_orig.append(
                        filename.split("\\")[-1].split('.')[0])
                    labels_orig.append(dir_name)
                else:
                    # 增强数据
                    imgs_agmt.append(img)  # 把图片数组加到一个列表里面
                    imgs_name_agmt.append(
                        filename.split("\\")[-1].split('.')[0])
                    labels_agmt.append(dir_name.split("_")[0])

        break

    end = time.clock()
    print("Cost time:", end-start, "s")
    print("Image shape:", imgs_orig[0].shape)
    print("Read ", len(labels_orig), " samples with ", len(
        labels_agmt), "augmentated in those classes:")
    tmp=""
    for key in classes:
        tmp+="'"+str(key)+"': "+str(labels_orig.count(str(key)))+" "
    print(tmp)
    print("")
    return imgs_orig, imgs_agmt, labels_orig, labels_agmt, classes, imgs_name_orig, imgs_name_agmt


def train_test_split(imgs_orig, labels_orig, imgs_agmt, labels_agmt, n_splits=3, now_splits=0, incl_agmt=False):
    """
    处理交叉检验中数据集分割问题
    将原始数据中每类数据按照同一百分比分割为n_splits份
    第now_splits份作为验证集，其余份作为训练集
    可为训练集读取增强数据

    Parameters
    ----------
    imgs_orig : list，（样本数），原始图片集，元素为（宽，高，通道）的图片numpy

    labels_orig : list，（样本数），原始图片名集，元素为分类名str

    imgs_agmt : list，（增强样本数），增强图片集，元素为（宽，高，通道）的图片numpy

    labels_agmt : list，（增强样本数），增强图片名集，元素为分类名str

    n_splits : int，数据集分割为n_splits份

    now_splits : int，取第now_splits份

    incl_agmt : bool，是否读取增强数据

    Returns
    -------
    imgs_train : numpy，（训练样本数，宽，高，通道），训练样本集

    imgs_test : numpy，（测试样本数，宽，高，通道），测试样本集

    labels_train : numpy，（训练样本数，），训练样本标签

    labels_test : numpy，（测试样本数，），训练样本标签

    Examples
    --------
       >>> imgs_train, imgs_test, labels_train, labels_test = io.train_test_split(imgs_orig, labels_orig, imgs_agmt, labels_agmt, n_splits, now_splits, incl_agmt)
    """

    print("--->Start spliting")
    start = time.clock()

    train_index, test_index = get_split_index(
        labels_orig, n_splits, now_splits)
    imgs_test = np.array(imgs_orig)[test_index]
    labels_test = np.array(labels_orig)[test_index]
    imgs_train = np.array(imgs_orig)[train_index]
    labels_train = np.array(labels_orig)[train_index]

    if incl_agmt:
        agmt_times = int(len(labels_agmt)/len(labels_orig))
        imgs_train = np.append(imgs_train, np.array(imgs_agmt)[
            [a+b for a in train_index*agmt_times for b in range(agmt_times)]], axis=0)
        labels_train = np.append(labels_train, np.array(labels_agmt)[
            [a+b for a in train_index*agmt_times for b in range(agmt_times)]], axis=0)

    end = time.clock()
    print("Cost time:", end-start, "s")
    print("Read the ", now_splits, "th part in ", n_splits, " parts")
    print("Train set size", len(labels_train),
          " with Test set size:", len(labels_test))
    print("")

    return imgs_train, imgs_test, labels_train, labels_test


def test_valid_split(imgs_test, labels_test, test_size=0.2, random_state=0):
    imgs_valid, imgs_test, labels_valid, labels_test = tts(
        imgs_test, labels_test, test_size=test_size, random_state=random_state)
    return imgs_valid, imgs_test, labels_valid, labels_test


def label_str2index(labels, classes):
    labels = np.vectorize(classes.get)(labels)
    return labels


def reverse_dict(dic):
    dic2 = dict(zip(dic.values(), dic.keys()))
    return dic2
