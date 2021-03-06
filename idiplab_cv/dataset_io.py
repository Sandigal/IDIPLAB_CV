# -*- coding: utf-8 -*-
"""
该模块 :meth:`dataset_io` 包含读取数据集以及数据集分割的类和函数。
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


class ShowProcess():
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
        process_bar_tmp.append("[" + "=" * num_arrow + ">")
        process_bar_tmp.append(". " * num_line + "] %5.2f%%" % percent)
        print("".join(process_bar_tmp), end="          ")
        if self.i >= self.max_steps:
            self.i = 0


def _get_split_index(labels, total_splits, valid_split):
    valid_split = valid_split if valid_split < total_splits else total_splits-1
    now_split = 0
    skf = StratifiedKFold(n_splits=total_splits)
    for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
        if now_split == valid_split:
            return train_index, test_index
        now_split += 1
    return None


def _read_imgs_in_dir(path, shape=None):
    names = os.listdir(path)
    imgs=[]
    if shape is None:
        imgs = [np.array(Image.open(path + "/"+name)) for name in names]
    else:
#        try:
        imgs = [np.array(Image.open(path + "/"+name).resize(shape, Image.ANTIALIAS)) for name in names]
#        except:
#            print("error")
##            for i in range(143):
##                Image.open(path + "/"+names[i])
    return imgs, names


def _read_imgs_in_dirs(path, dirs_name, shape=None):
    imgall = []
    labelall = []
    nameall = []

    sub_dir_list = os.listdir(path+"/origin")
    process_bar = ShowProcess(len(sub_dir_list))
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


def label_smooth(labels, section=None):
    """标签平滑

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    :param numpy path: An open Bigtable Table instance.
    :param numpy section: An open Bigtable Table instance.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {"Serak": ("Rigel VII", "Preparer"),
         "Zim": ("Irk", "Invader"),
         "Lrrr": ("Omicron Persei 8", "Emperor")}

        If a key from the keys argument is missing from the dictionary,
        then that row was not found in the table.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.
    """

    labels = np.copy(labels)
    labels = labels*1.0
    eps = 0.1
    for i in range(1, len(section)):
        smooth_num = section[i]-section[i-1]
        labels_reference = labels[:, section[i-1]:section[i]]
        labels_validd = np.sum(labels_reference, axis=1)
        labels_minuend = labels_reference * (1 - eps)
        labels_added = np.tile(
            (eps * labels_validd / smooth_num).reshape(len(labels_validd), 1), (1, smooth_num))
        labels[:, section[i-1]:section[i]] = labels_minuend+labels_added
    return labels

# %%


class Dataset(object):
    """Class methods are similar to regular functions.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Args:
        augment (:obj:`bool`, 可选): 是否读取增强数据。

    Attributes:
        imgs_origin (str): Description of `attr1`.
        labels_origin (:obj:`int`, optional): Description of `attr2`.
        names_origin (str): Description of `attr1`.
        imgs_augment (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, augment=False):
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
        self.test_index = []
        self.valid_index = []
        self.augment_index = []
        self.augment_cross_index = []

    def _load_file(self, path, shape=None, step=1):

        if step == 1:
            self.imgs_origin, self.labels_origin, self.names_origin = _read_imgs_in_dirs(
                path, "/origin/", shape)

            if self.augment:
                self.imgs_augment, self.labels_augment, self.names_augment = _read_imgs_in_dirs(
                    path, "/augment/", shape)
        else:
            self.imgs_origin, self.labels_origin, self.names_origin = _read_imgs_in_dirs(
                path, "/crop_AB/", shape)

            if self.augment:
                self.imgs_augment, self.labels_augment, self.names_augment = _read_imgs_in_dirs(
                    path, "/crop_SC_augment/", shape)

        sub_dir_list = os.listdir(path+"/origin")
        self.class_to_index = dict(zip(sub_dir_list, range(len(sub_dir_list))))

    def _load_h5(self, path):
        file = open(path, "rb")
        contact = load(file)
        self.imgs_origin = contact["imgs_origin"]
        self.labels_origin = contact["labels_origin"]
        if self.augment is True:
            self.imgs_augment = contact["imgs_augment"]
            self.labels_augment = contact["labels_augment"]
            self.names_augment = contact["names_augment"]
        self.names_origin = contact["names_origin"]
        self.class_to_index = contact["class_to_index"]
        self.sample_per_class = contact["sample_per_class"]
        file.close()

    def load_data(self, path, shape=None):
        """读取图像

        支持两种读取模式，函数会自动从 ``path`` 中进行判断。
            1. 从文件夹中读取所有jpg、png等图片格式的文件。目录结构请参照 :ref:`目录结构`。
            2. 从打包的 :obj:`h5` 文件中读取数据。您需要先使用方法1进行读取

        Note:
            运行方法2的前提是已经有一个打包的:obj:`h5` 文件。为此，需要先用方法1读取数据，再用 :meth:`create_h5` 打包。

        Args:
            path (:obj:`str`): 数据所在目录地址，或 :obj:`h5` 文件地址。
            shape (:obj:`turple` of :obj:`int`, 可选): 所有的图像都将以该尺寸读入内存。格式为 `(width, height)`，默认为`(336, 224)`。
            augment (:obj:`bool`, 可选): 是否读取增强数据。

        Returns:
                class_to_index (:obj:`dict` of :obj:`str` to :obj:`int`): 各类对应的标签序号。
                sample_per_class (:obj:`dict` of :obj:`str` to :obj:`int`): 各类对应的样本数量。

        Examples:

            >>> dataset = io.Dataset(augment=False)
            >>> class_to_index, sample_per_class = dataset.load_data(
            ...         path="../data/dataset 336x224",
            ...         shape=(336, 224))
            --->Start loading data
            -->Processing for C3 [=============================>] 100.00%
            Cost time: 2.702s
            Image shape (hight, width, channel): (224, 336, 3)
            Read 400 samples
            Class index: {'C1': 0, 'C2': 1, 'C3': 2}
            Sample per class: {'C3': 120, 'C2': 60, 'C1': 220}

            >>> dataset = io.Dataset(augment=True)
            >>> class_to_index, sample_per_class = dataset.load_data(
            ...         path="dataset_test.h5")
            --->Start loading data
            Cost time: 1.132s
            Image shape (hight, width, channel): (224, 336, 3)
            Read 400 samples with  800 augmentated
            Class index: {'C1': 0, 'C2': 1, 'C3': 2}
            Sample per class: {'C3': 360, 'C2': 180, 'C1': 660}

        """

        print("--->Start loading data ")
        start = time.clock()

        if path.endswith("h5") is False:
            self._load_file(path, shape)
            print("")
        else:
            self._load_h5(path)

        classes = set(self.labels_origin)
        for classe in classes:
            self.sample_per_class[classe] = self.labels_origin.count(classe)
        if self.augment is True:
            self.sample_per_class[classe] += self.labels_augment.count(classe)

        end = time.clock()
        print("Cost time: %.3fs" % (end-start))
        print("Image shape (hight, width, channel):",
              self.imgs_origin[0].shape)
        print("Read", len(self.labels_origin), "samples (%.2fMB)" %
              (np.array(self.imgs_origin).nbytes / (1024 * 1000.0)), end="")
        if self.augment:
            print(" with", len(self.labels_augment), "augmentated (%.2fMB)" %
                  (np.array(self.imgs_augment).nbytes / (1024 * 1000.0)), end="")
        print("")
        print("Class index: "+str(self.class_to_index))
        print("Sample per class: "+str(self.sample_per_class))
        print("")

        return self.class_to_index, self.sample_per_class

    def create_h5(self, name):
        """Generators have a ``Yields`` section instead of a ``Returns`` section.

        Args:
            n (int): The upper limit of the range to generate, from 0 to `n` - 1.

        Returns:
            bool: True if successful, False otherwise.

            The return type is optional and may be specified at the beginning of
            the ``Returns`` section followed by a colon.

            The ``Returns`` section may span multiple lines and paragraphs.
            Following lines should be indented to match the first line.

            The ``Returns`` section supports any reStructuredText formatting,
            including literal blocks ::

                {
                    'param1': param1,
                    'param2': param2
                }

        Examples:
            Examples should be written in doctest format, and should illustrate how
            to use the function.

            >>> print([i for i in example_generator(4)])
            [0, 1, 2, 3]

        """
        content = {"imgs_origin": self.imgs_origin,
                   "imgs_augment": self.imgs_augment,
                   "labels_origin": self.labels_origin,
                   "labels_augment": self.labels_augment,
                   "names_origin": self.names_origin,
                   "names_augment": self.names_augment,
                   "class_to_index": self.class_to_index,
                   "sample_per_class": self.sample_per_class}
        file = open(name, "wb")
        dump(content, file, True)
        file.close()

    def train_test_split(self, total_splits=5, test_split=0):
        print("--->Start spliting dataset to trainset and testset")
        start = time.clock()

        self.train_index, self.test_index = _get_split_index(
            self.labels_origin, total_splits, test_split)
        imgs_train = np.array(self.imgs_origin, dtype="float32")[
            self.train_index]
        labels_train = np.array(self.labels_origin)[
            self.train_index]
        imgs_test = np.array(self.imgs_origin, dtype="float32")[self.test_index]
        labels_test = np.array(self.labels_origin)[self.test_index]

        if self.augment is True and len(self.labels_augment) > 0:
            augment_amount = int(
                len(self.labels_augment)/len(self.labels_origin))
            self.augment_index = [a+b
                                  for a in self.train_index * augment_amount
                                  for b in range(augment_amount)]
            imgs_train = np.append(imgs_train, np.array(
                self.imgs_augment, dtype="float32")[self.augment_index], axis=0)
            labels_train = np.append(labels_train, np.array(
                self.labels_augment)[self.augment_index], axis=0)

        end = time.clock()
        print("Cost time: %.3fs" % (end-start))
        print("The", test_split, "th split in", total_splits, "splits is testset")
        print("Train set size", len(labels_train), "(%.2fMB)" % (imgs_train.nbytes / (1024 * 1000.0)),
              "with Test set size", len(labels_test), "(%.2fMB)" % (imgs_test.nbytes / (1024 * 1000.0)))
        print("")

        return imgs_train, labels_train, imgs_test, labels_test

    def cross_split(self, total_splits=3, valid_split=0):
        print("--->Start spliting trainset to subtrainset and validset")
        start = time.clock()

        labels_train = np.array(self.labels_origin)[
            self.train_index].tolist()
        self.train_cross_index, self.valid_index = _get_split_index(
            labels_train, total_splits, valid_split)
        imgs_train = np.array(self.imgs_origin, dtype="float32")[
            self.train_index][self.train_cross_index]
        labels_train = np.array(self.labels_origin)[
            self.train_index][self.train_cross_index]
        imgs_valid = np.array(self.imgs_origin, dtype="float32")[
            self.train_index][self.valid_index]
        labels_valid = np.array(self.labels_origin)[
            self.train_index][self.valid_index]
        names_valid = np.array(self.names_origin)[
            self.train_index][self.valid_index]

        if self.augment is True and len(self.labels_augment) > 0:
            augment_amount = int(
                len(self.labels_augment)/len(self.labels_origin))
            self.augment_cross_index = [
                a+b
                for a in self.train_cross_index * augment_amount
                for b in range(augment_amount)]
            imgs_train = np.append(
                imgs_train,
                np.array(self.imgs_augment, dtype="float32")[
                    self.augment_index][self.augment_cross_index],
                axis=0)
            labels_train = np.append(
                labels_train,
                np.array(self.labels_augment)[
                    self.augment_index][self.augment_cross_index],
                axis=0)

        end = time.clock()
        print("Cost time: %.3fs" % (end-start))
        print("The", valid_split, "th split in", total_splits, "splits is validset")
        print("Trainset size", len(labels_train), "(%.2fMB)" % (imgs_train.nbytes / (1024 * 1000.0)),
              "with validset size", len(labels_valid), "(%.2fMB)" % (imgs_valid.nbytes / (1024 * 1000.0)))
        print("")
        return imgs_train, labels_train, imgs_valid, labels_valid, names_valid

    def VOC_Indexes(self, category=""):
        """产生VOC格式的文件索引

        """
        print("--->Start spliting trainset to subtrainset and validset")
        start = time.clock()
        file_prefix="" if category=="" else "_"

        names_train = np.array(self.names_origin)[self.train_index]
        f=open(category+file_prefix+"train.txt", "a+")
        for name_train in names_train:
            name_train = name_train[:name_train.rfind('.')]+ '\n'
            f.write(name_train)
        f.close()

        names_test = np.array(self.names_origin)[self.test_index]
        f=open(category+file_prefix+"test.txt", "a+")
        for name_test in names_test:
            name_test = name_test[:name_test.rfind('.')]+ '\n'
            f.write(name_test)
        f.close()

        end = time.clock()
        print("Cost time: %.3fs" % (end-start))
        print("Creat", str(len(names_train)), "lines in", category+file_prefix+"trainval.txt at", os.path.curdir)
        print("Creat", str(len(names_test)), "lines in", category+file_prefix+"test.txt in", os.path.curdir)
        print("")

