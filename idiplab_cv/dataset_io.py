"""
该模块:`dataset_io`包含读取数据集以及数据集分割的类和函数。
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0

import os
from pickle import dump
from pickle import load
import time

from keras.utils import to_categorical
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle


class _ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """

    def __init__(self, max_steps):
        """
        初始化函数，需要知道总共的处理次数
        """
        self.max_steps = max_steps
        self.i = 0
        self.max_arrow = 29

    def show_process(self, name=None, i=None):
        """
        显示函数，根据当前的处理进度显示进度
        效果为->Processing for P5 [=============================>]100.00%
        """
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = self.i * self.max_arrow // self.max_steps  # 计算显示多少个"="
        num_line = self.max_arrow - num_arrow  # 计算显示多少个". "
        percent = self.i * 100.0 / self.max_steps  # 计算完成进度，格式为xx.xx%
        process_bar_tmp = ["\r"]
        if name is not None:
            process_bar_tmp.append("-->Processing for %s " % name)
        process_bar_tmp.append(
            "[" + "=" * num_arrow + ">"+". " * num_line + "] %5.2f%%" % percent)
        print("".join(process_bar_tmp),end="          ")
        if self.i >= self.max_steps:
            self.i = 0

def _get_split_index(labels, total_splits, valid_split):
    now_split = 0
    skf = StratifiedKFold(n_splits=total_splits)
    for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
        if now_split == valid_split:
            return train_index, test_index
        now_split += 1


def _read_imgs_in_dir(path, shape=None):
    names = os.listdir(path)
    if shape is None:
        imgs = [np.array(Image.open(path + "/"+name)) for name in names]
    else:
        imgs = [np.array(Image.open(path + "/"+name).resize(shape,
                                                            Image.ANTIALIAS)) for name in names]
    return imgs, names


def _read_imgs_in_dirs(path, dirs_name, shape=(224, 224)):
    imgall = []
    labelall = []
    nameall = []

    sub_dir_list = os.listdir(path+"/origin")
    process_bar = _ShowProcess(len(sub_dir_list))
    for sub_dir in sub_dir_list:
        process_bar.show_process(sub_dir)

        imgs, names = _read_imgs_in_dir(path+dirs_name+sub_dir, shape)
        imgall.extend(imgs)
        labelall.extend([sub_dir]*len(imgs))
        nameall.extend(names)
    imgs = np.array(imgs)
    return imgall, labelall, nameall


def label_str2index(labels, classes):
    labels = np.vectorize(classes.get)(labels)
    return labels


def reverse_dict(dic):
    dic2 = dict(zip(dic.values(), dic.keys()))
    return dic2


def label_smooth(labels, section):
    """标签平滑

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    参数:
        labels: An open Bigtable Table instance.
        section: A sequence of strings representing the key of each table row
            to fetch.

    返回:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {"Serak": ("Rigel VII", "Preparer"),
         "Zim": ("Irk", "Invader"),
         "Lrrr": ("Omicron Persei 8", "Emperor")}

        If a key from the keys argument is missing from the dictionary,
        then that row was not found in the table.

    错误:
        IOError: An error occurred accessing the bigtable.Table object.
    """
    labels = np.copy(labels)
    eps = 0.1
    for i in range(len(labels)):
        for j in range(1, len(section)):
            if np.argmax(labels[i]) < section[j]:
                smooth_num = section[j]-section[j-1]
                labels[i][section[j-1]:section[j]] = labels[i][section[j-1]:section[j]] * \
                    (1 - eps)+(1-labels[i][section[j-1]
                     :section[j]]) * eps / (smooth_num)
                break
    return labels

# %%


class Dataset(object):
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self,
                 path,
                 shape=(224, 224),
                 augment=False):
        self.path = path
        self.shape = shape
        self.augment = augment
        self.imgs_origin = []
        self.labels_origin = []
        self.names_origin = []
        self.imgs_augment = []
        self.labels_augment = []
        self.names_augment = []
        self.class_to_index = {}
        self.sample_per_class = {}
        self.train_index = []
        self.train_cross_index = []
        self.augment_index = []
        self.augment_cross_index = []

    def load_data(self, step=1):
        print("--->Start loading data ")
        start = time.clock()

        if step == 1:
            self.imgs_origin, self.labels_origin, self.names_origin = _read_imgs_in_dirs(
                self.path, "/origin/", self.shape)

            if self.augment:
                self.imgs_augment, self.labels_augment, self.names_augment = _read_imgs_in_dirs(
                    self.path, "/augment/", self.shape)
        else:
            self.imgs_origin, self.labels_origin, self.names_origin = _read_imgs_in_dirs(
                self.path, "/crop_AB/", self.shape)

            if self.augment:
                self.imgs_augment, self.labels_augment, self.names_augment = _read_imgs_in_dirs(
                    self.path, "/crop_SC_augment/", self.shape)

        sub_dir_list = os.listdir(self.path+"/origin")
        self.class_to_index = dict(zip(sub_dir_list, range(len(sub_dir_list))))

        classes = set(self.labels_origin)
        for classe in classes:
            self.sample_per_class[classe] = self.labels_origin.count(classe)

        end = time.clock()
        print("")
        print("Cost time: %.3fs" % (end-start))
        print("Image shape (hight, width, channel):",
              self.imgs_origin[0].shape)
        print("Read", len(self.labels_origin), "samples", end="")
        if self.augment:
            print(" with ", len(self.labels_augment), "augmentated", end="")
        print("")
        print("Class index: "+str(self.class_to_index))
        print("Sample per class: "+str(self.sample_per_class))
        print("")
        return self.class_to_index, self.sample_per_class

    def create_h5(self, name):
        content = {
            "imgs_origin": self.imgs_origin,
            "imgs_augment": self.imgs_augment,
            "labels_origin": self.labels_origin,
            "labels_augment": self.labels_augment,
            "names_augment": self.names_augment,
            "class_to_index": self.class_to_index,
            "sample_per_class": self.sample_per_class,
        }
        f = open(name, "wb")
        dump(content, f, True)
        f.close()

    def load_h5(self, name):
        f = open(name, "rb")
        contact = load(f)
        self.imgs_origin = contact["imgs_origin"]
        self.imgs_augment = contact["imgs_augment"]
        self.labels_origin = contact["labels_origin"]
        self.labels_augment = contact["labels_augment"]
        self.names_augment = contact["names_augment"]
        self.class_to_index = contact["class_to_index"]
        self.sample_per_class = contact["sample_per_class"]
        self.imgs_origin = contact["imgs_origin"]
        self.imgs_origin = contact["imgs_origin"]
        self.imgs_origin = contact["imgs_origin"]
        f.close()

        print("")
        print("Image shape:", self.imgs_origin[0].shape)
        print("Read", len(self.labels_origin), "samples", end="")
        if self.augment:
            print(" with ", len(self.labels_augment), "augmentated", end="")
        print("")
        print("Class index: "+str(self.class_to_index))
        print("Sample per class: "+str(self.sample_per_class))
        print("")

    def train_test_split(self, test_shape=0.25):
        print("--->Start spliting dataset to trainset and testset")
        start = time.clock()

        n_splits = int(1/test_shape)

        self.train_index, test_index = _get_split_index(
            self.labels_origin, n_splits, 0)
        imgs_train = np.array(self.imgs_origin)[self.train_index]
        labels_train = np.array(self.labels_origin)[self.train_index]
        imgs_test = np.array(self.imgs_origin)[test_index]
        labels_test = np.array(self.labels_origin)[test_index]

        if self.augment:
            augment_amount = int(
                len(self.labels_augment)/len(self.labels_origin))
            self.augment_index = [a+b for a in self.train_index *
                                  augment_amount for b in range(augment_amount)]
            imgs_train = np.append(imgs_train, np.array(
                self.imgs_augment)[self.augment_index], axis=0)
            labels_train = np.append(labels_train, np.array(
                self.labels_augment)[self.augment_index], axis=0)

        end = time.clock()
        print("Cost time: %.3fs" % (end-start))
        print("In fact the first %d%% of data are test set" % (100/n_splits))
        print("Train set size", len(labels_train),
              "with Test set size", len(labels_test))
        print("")

        return imgs_train, labels_train, imgs_test, labels_test

    def cross_split(self, total_splits=3, valid_split=0):
        print("--->Start spliting trainset to subtrainset and validset")
        start = time.clock()

        labels_train = np.array(self.labels_origin)[self.train_index].tolist()
        self.train_cross_index, valid_index = _get_split_index(
            labels_train, total_splits, valid_split)
        imgs_train = np.array(self.imgs_origin)[
            self.train_index][self.train_cross_index]
        labels_train = np.array(self.labels_origin)[
            self.train_index][self.train_cross_index]
        imgs_valid = np.array(self.imgs_origin)[
            self.train_index][valid_index]
        labels_valid = np.array(self.labels_origin)[
            self.train_index][valid_index]

        if self.augment:
            augment_amount = int(
                len(self.labels_augment)/len(self.labels_origin))
            self.augment_cross_index = [a+b for a in self.train_cross_index *
                                        augment_amount for b in range(augment_amount)]
            imgs_train = np.append(imgs_train, np.array(self.imgs_augment)[
                                   self.augment_index][self.augment_cross_index], axis=0)
            labels_train = np.append(labels_train, np.array(self.labels_augment)[
                                     self.augment_index][self.augment_cross_index], axis=0)

        end = time.clock()
        print("Cost time: %.3fs" % (end-start))
        print("The", valid_split, "th split in",
              total_splits, "splits is validset")
        print("Trainset size", len(labels_train),
              "with validset size", len(labels_valid))
        print("")
        return imgs_train, labels_train, imgs_valid, labels_valid
