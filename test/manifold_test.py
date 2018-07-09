# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:03:03 2018

@author: Sandiagal
"""

import dataset_io as io
import manifold
from time import time

import numpy as np


#%%

path = "../data/tmp/"
params, classToIndex, samplePerClass = io.load_all_data(path, shape=(150, 150))
imgs_origin = np.array(params["origin"]["imgs"])
labels_origin = np.array(params["origin"]["labels"])
labels_origin = io.label_str2index(labels_origin, classToIndex)

imgs = imgs_origin
labels = labels_origin
manifold_args = dict(
    Random=True,
    RandomTrees=True,
    MDS=True,
    PCA=True,
    #    LinearDiscriminant=False,
    #    Isomap=False,
    #    Spectral=False,
    #    LLE=False,
    #    ModifiedLLE=False,
    #    HLLE=False,
    #    LTSA=False,
    #    TSNE=False,
    n_neighbors=30)

manifold.manifold(imgs, labels, manifold_args)