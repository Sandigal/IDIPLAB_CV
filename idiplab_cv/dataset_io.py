# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:02:44 2018

@author: Sandiagal
"""

from PIL import Image
import os
import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical

# %%


def get_split_index(labels, n_splits, now_split):
    i = 0
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
        #        print("TRAIN:", train_index, "TEST:", test_index)
        if i == now_split:
            return train_index, test_index
        i = i+1


def read_imgs_in_dir(path, shape=(224, 224)):
    names = os.listdir(path)
    imgs = [np.array(Image.open(path + "/"+name).resize(shape,
                                                        Image.ANTIALIAS)) for name in names]
    return imgs, names


def read_imgs_in_dirs(path, dirs_name, shape=(224, 224)):
    imgall = []
    labelall = []
    nameall = []
    sub_dir_list = os.listdir(path+"/origin")
    for sub_dir in sub_dir_list:
        imgs, names = read_imgs_in_dir(path+dirs_name+sub_dir, shape)
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

# %%


class Dataset(object):

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
        self.classToIndex = {}
        self.samplePerClass = {}
        self.train_index = []
        self.train_cross_index = []
        self.augment_index = []
        self.augment_cross_index = []

    def load_data(self):
        print("--->Start loadind data ")
        start = time.clock()

        self.imgs_origin, self.labels_origin, self.names_origin = read_imgs_in_dirs(
            self.path, "/origin/", self.shape)

        if self.augment:
            self.imgs_augment, self.labels_augment, self.names_augment = read_imgs_in_dirs(
                self.path, "/augment/", self.shape)

        sub_dir_list = os.listdir(self.path+"/origin")
        self.classToIndex = dict(zip(sub_dir_list, range(len(sub_dir_list))))

        classes = set(self.labels_origin)
        for classe in classes:
            self.samplePerClass[classe] = self.labels_origin.count(classe)

        end = time.clock()
        print("Cost time: %.3fs" % (end-start))
        print("Image shape:", self.imgs_origin[0].shape)
        if self.augment:
            print("Read", len(self.labels_origin), "samples with ", len(
                self.labels_augment), "augmentated")
        else:
            print("Read", len(self.labels_origin), "samples")
        print("Class index: "+str(self.classToIndex))
        print("Sample per class: "+str(self.samplePerClass))
        print("")
        return self.classToIndex, self.samplePerClass

    def train_test_split(self, test_shape=0.25):
        print("--->Start spliting dataset to trainset and testset")
        start = time.clock()

        n_splits = int(1/test_shape)

        self.train_index, test_index = get_split_index(
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

    def cross_split(self, total_splits=3, now_split=0):
        print("--->Start spliting trainset to subtrainset and validset")
        start = time.clock()

        labels_train = np.array(self.labels_origin)[self.train_index].tolist()
        self.train_cross_index, valid_index = get_split_index(
            labels_train, total_splits, now_split)
        imgs_train = np.array(self.imgs_origin)[
            self.train_index][self.train_cross_index]
        labels_train = np.array(self.labels_origin)[
            self.train_index][self.train_cross_index]
        imgs_valid = np.array(self.imgs_origin)[
            self.train_index][self.train_cross_index]
        labels_valid = np.array(self.labels_origin)[
            self.train_index][self.train_cross_index]

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
        print("The", now_split, "th split in",
              total_splits, "splits is validset")
        print("Trainset size", len(labels_train),
              "with validset size", len(labels_valid))
        print("")
        return imgs_train, labels_train, imgs_valid, labels_valid
