# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:03:03 2018

@author: Sandiagal
"""

from time import time

import dataset_io as io
import manifold

#%%

dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
        path="../FOB valid 298x224",
        shape=(99, 66))

imgs_origin = dataset.imgs_origin
labels_origin = dataset.labels_origin
labels_origin = io.label_str2index(labels_origin, class_to_index)

imgs = imgs_origin
labels = labels_origin

#%%
import manifold
manifold_args = dict(
#    Random=True,
#        RandomTrees=True,
#        MDS=True,
#        PCA=True,
#        LinearDiscriminant=True,
#        Isomap=False,
#        Spectral=True,
#        LLE=False,
#        ModifiedLLE=False,
#        HLLE=False,
#        LTSA=False,
        TSNE=True,
    n_neighbors=30)

manifold.manifold(imgs, labels, manifold_args,showLabels=False, showImages=False)