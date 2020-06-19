"""
该模块:`XXX`包含XXX的类和函数。
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0

import models


# %%


input_shape = (None, None, 3)  # 32的倍数
model = models.xceptionGAP(input_shape,classes=2, include_top=False)
model.summary()

#
# %%


import dataset_io as io
import numpy as np

dataset = io.Dataset(augment=False)
class_to_index, sample_per_class = dataset.load_data(
    path="dataset_1_1_origin.h5")

labels_origin = np.array(dataset.labels_origin)
labels_origin = io.label_str2index(labels_origin, class_to_index)
imgs_origin = np.array(dataset.imgs_origin)

# %%

scores_predict=[]
for img in imgs:
    img=np.expand_dims(img, axis=0)
    score_predict = model.predict(
            img,
            verbose=1)
    scores_predict.append(score_predict.flatten())

imgs_f = np.array(scores_predict)
# %%

#from pickle import dump
# result = {
#    "scores_predict": scores_predict,
# }
#f = open("scores_predict.h5", "wb")
#dump(result, f, True)
# f.close()

# %%

#from pickle import load
#
#f = open("scores_predict.h5", "rb")
#contact = load(f)
#scores_predict = contact["scores_predict"]
# f.close()


# %%

#import manifold
#
# manifold_args = dict(
#    RandomTrees=True,
#    PCA=True,
#    LinearDiscriminant=True,
#    Spectral=True,
#    TSNE=True,
#    n_neighbors=30)
#
#index_to_class = io.reverse_dict(class_to_index)
#
# manifold.manifold(imgs_origin, labels_origin, manifold_args, scores_predict,
#                  index_to_class=index_to_class, showLabels=False, showImages=False,
#                  imageZoom=0.15, imageDist=8e-3)


# %%

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from collections import Counter

print(sorted(Counter(labels_origin).items()))

smote_enn = TomekLinks(random_state=0)
scores_predict2, labels_origin2 = smote_enn.fit_sample(
    scores_predict, labels_origin)
print(sorted(Counter(labels_origin2).items()))

# %%

import manifold

manifold_args = dict(
    RandomTrees=True,
    PCA=True,
    LinearDiscriminant=True,
    Spectral=True,
    TSNE=True,
    n_neighbors=30)

index_to_class = io.reverse_dict(class_to_index)

manifold.manifold(imgs_origin, labels_origin2, manifold_args, scores_predict2,
                  index_to_class=index_to_class, showLabels=False, showImages=False,
                  imageZoom=0.15, imageDist=8e-3)
