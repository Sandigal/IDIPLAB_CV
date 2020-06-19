# -*- coding: utf-8 -*-
"""
流形学习
"""

# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0

import os
from time import time
import shutil

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# %%


def tensorboard_embed(LOG_DIR, imgs_test, labels_test, scores_predict):
    #    LOG_DIR = 'E:\Temporary\DeapLeraning\Medical\logs'  # FULL PATH HERE!!!

    # build a metadata file with the real labels of the test set
    metadata_file = os.path.join(LOG_DIR, 'metadata.tsv')
    with open(metadata_file, 'w') as f:
        for i in range(len(labels_test)):
            f.write('{}\n'.format(labels_test[i]))

    # get the sprite image mnist_10k_sprite.png as provided by the TensorFlow guys here, and place it in your LOG_DIR
    max_x = int(336/2)
    max_y = int(224/2)
    x_test = np.array([np.array(Image.fromarray(np.uint8(row)).resize(
        (max_x, max_y), Image.ANTIALIAS).crop(((max_x-max_y)/2, 0, (max_x-max_y)/2+max_y, max_y))) for row in imgs_test])
    img_array = np.expand_dims(x_test, axis=1)
    img_array = img_array.reshape((17, 5, max_y, max_y, 3))
    # Image.fromarray(np.uint8(img_array[0][0])).show()
    img_array_flat = np.concatenate(
        [np.concatenate([x for x in row], axis=1) for row in img_array])
    img = Image.fromarray(np.uint8(255-img_array_flat))
    img.save(os.path.join(LOG_DIR, 'sprite_images.jpg'))

    # write some Tensorflow code:
    embedding_var = tf.Variable(scores_predict,  name='final_layer_embedding')
    sess = tf.Session()
    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify the metadata file:
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    # Specify the sprite image:
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_images.jpg')
    embedding.sprite.single_image_dim.extend(
        [max_y, max_y])  # image size = 28x28

    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, os.path.join(LOG_DIR, 'model2.ckpt'), 1)

    is_exist = os.path.exists(LOG_DIR+'/logs')
    if not is_exist:
        os.makedirs(LOG_DIR+'/logs')
    shutil.move(LOG_DIR+"/metadata.tsv", LOG_DIR+"/logs/metadata.tsv")
    shutil.move(LOG_DIR+"/sprite_images.jpg",
                LOG_DIR+"/logs/sprite_images.jpg")


def plot_embedding(X, labels, imgs, title=None, **kwargs):
    # Scale and visualize the embedding vectors
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(16, 9))
    ax = plt.subplot(111)

    curLabel = -1
    for i in range(X.shape[0]):
        if kwargs['showLabels'] is True:
            plt.text(X[i, 0], X[i, 1], str(kwargs['index_to_class'][labels[i]]),
                     color=plt.cm.Set1(labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': kwargs['fontSize']})
        else:
            ps = plt.scatter(X[i, 0], X[i, 1],
                             s=kwargs['area'],
                             color=plt.cm.Set1(labels[i] / 6.))
            if curLabel != labels[i]:
                curLabel = labels[i]
                ps.set_label(kwargs['index_to_class'][labels[i]])

    if kwargs['showImages'] is True:
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(len(X)):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < kwargs['imageDist']:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(
                        imgs[i], zoom=kwargs['imageZoom'], cmap=plt.cm.gray_r),
                    X[i], pad=.0)
                ax.add_artist(imagebox)

    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="best", fontsize=20, shadow=1)
    plt.savefig(title.split("(")[0].strip())


def Random(X, labels, imgs, **kwargs):
    # Random 2D projection using a random unitary matrix
    print("Computing random projection")
    t = time()
    rp = SparseRandomProjection(
        n_components=2, random_state=0)
    X_projected = rp.fit_transform(X)
    plot_embedding(X_projected, labels, imgs, "Random Projection of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def RandomTrees(X, labels, imgs, **kwargs):
    """
    稳定

    """
    # Random Trees embedding of the dataset dataset
    print("Computing Random Trees embedding")
    hasher = RandomTreesEmbedding(n_estimators=200, random_state=0,
                                  max_depth=5)
    t = time()
    X_transformed = hasher.fit_transform(X)
    pca = TruncatedSVD(n_components=2)
    X_reduced = pca.fit_transform(X_transformed)

    plot_embedding(X_reduced, labels, imgs,
                   "Random Trees embedding of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def mds(X, labels, imgs, **kwargs):
    # MDS  embedding of the dataset dataset
    print("Computing MDS embedding")
    clf = MDS(n_components=2, n_init=1, max_iter=100)
    t = time()
    X_mds = clf.fit_transform(X)
    print("Done. Stress: %f" % clf.stress_)
    plot_embedding(X_mds, labels, imgs,
                   "MDS embedding of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def PCA(X, labels, imgs, **kwargs):
    """
    稳定

    """
    # Projection on to the first 2 principal components
    print("Computing Principal Components projection")
    t = time()
    X_pca = TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca, labels, imgs,
                   "Principal Components projection of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def LinearDiscriminant(X, labels, imgs, **kwargs):
    """
    稳定

    """
    # projection on to the first 2 linear discriminant components
    print("Computing Linear Discriminant Analysis projection")
    t = time()
    X2 = X.copy()
    X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
    t = time()
    X_lda = LinearDiscriminantAnalysis(
        n_components=2).fit_transform(X2, labels)
    plot_embedding(X_lda, labels, imgs,
                   "Linear Discriminant projection of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def isomap(X, labels, imgs, n_neighbors, **kwargs):
    # Isomap projection of the dataset dataset
    print("Computing Isomap embedding")
    t = time()
    X_iso = Isomap(n_neighbors, n_components=2).fit_transform(X)
    print("Done.")
    plot_embedding(X_iso, labels, imgs,
                   "Isomap projection of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def Spectral(X, labels, imgs, **kwargs):
    """
    稳定

    """
    # Spectral embedding of the dataset dataset
    print("Computing Spectral embedding")
    embedder = SpectralEmbedding(n_components=2, random_state=0,
                                 eigen_solver="arpack")
    t = time()
    X_se = embedder.fit_transform(X)

    plot_embedding(X_se, labels, imgs,
                   "Spectral embedding of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def LLE(X, labels, imgs, n_neighbors, **kwargs):
    # Locally linear embedding of the dataset dataset
    print("Computing Locally Linear embedding")
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                 method='standard')
    t = time()
    X_lle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_lle, labels, imgs,
                   "Locally Linear Embedding of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def ModifiedLLE(X, labels, imgs, n_neighbors, **kwargs):
    # Modified Locally linear embedding of the dataset dataset
    print("Computing modified LLE embedding")
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                 method='modified')
    t = time()
    X_mlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_mlle, labels, imgs,
                   "Modified Locally Linear Embedding of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def HLLE(X, labels, imgs, n_neighbors, **kwargs):
    # Hessian Locally Linear embedding of the dataset dataset
    print("Computing Hessian LLE embedding")
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                 method='hessian')
    t = time()
    X_hlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_hlle, labels, imgs,
                   "Hessian Locally Linear Embedding of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def LTSA(X, labels, imgs, n_neighbors, **kwargs):
    # LTSA embedding of the dataset dataset
    print("Computing LTSA embedding")
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                 method='ltsa')
    t = time()
    X_ltsa = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_ltsa, labels, imgs,
                   "LTSA of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def tsne(X, labels, imgs, **kwargs):
    """
    稳定

    """
    # t-SNE embedding of the dataset dataset
    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne, labels, imgs,
                   "t-SNE embedding of the dataset (time %.2fs)" %
                   (time() - t), **kwargs)


def manifold(imgs, labels, manifold_args, featrue=None, **kwargs):

    imgs = np.copy(imgs)

    if featrue is None:
        #    visul.overall(imgs, 64)
        featrue = imgs.reshape((len(imgs), -1))
        featrue = featrue/255

    kwdefaults = {
        'index_to_class': dict(zip(range(len(set(labels))), set(labels))),
        'showLabels': True,
        'fontSize': 50,
        'area': 100,
        'showImages': True,
        'imageZoom': 0.8,
        'imageDist': 4e-3}

    allowedkwargs = [
        'index_to_class',
        'showLabels',
        'fontSize',
        'area',
        'showImages',
        'imageZoom',
        'imageDist']

    for key in kwargs:
        if key not in allowedkwargs:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowedkwargs))

    # Set kwarg defaults
    for kw in allowedkwargs:
        kwargs.setdefault(kw, kwdefaults[kw])

    if 'Random' in manifold_args and manifold_args['Random']:
        Random(featrue, labels, imgs, **kwargs)
    if 'RandomTrees' in manifold_args and manifold_args['RandomTrees']:
        RandomTrees(featrue, labels, imgs, **kwargs)
    if 'MDS' in manifold_args and manifold_args['MDS']:
        mds(featrue, labels, imgs, **kwargs)
    if 'PCA' in manifold_args and manifold_args['PCA']:
        PCA(featrue, labels, imgs, **kwargs)
    if 'LinearDiscriminant' in manifold_args and manifold_args['LinearDiscriminant']:
        LinearDiscriminant(featrue, labels, imgs, **kwargs)
    if 'Isomap' in manifold_args and manifold_args['Isomap'] and 'n_neighbors' in manifold_args:
        isomap(featrue, labels, imgs, manifold_args['n_neighbors'], **kwargs)
    if 'Spectral' in manifold_args and manifold_args['Spectral']:
        Spectral(featrue, labels, imgs, **kwargs)
    if 'LLE' in manifold_args and manifold_args['LLE'] and 'n_neighbors' in manifold_args:
        LLE(featrue, labels, imgs, manifold_args['n_neighbors'], **kwargs)
    if 'ModifiedLLE' in manifold_args and manifold_args['ModifiedLLE'] and 'n_neighbors' in manifold_args:
        ModifiedLLE(featrue, labels, imgs,
                    manifold_args['n_neighbors'], **kwargs)
    if 'HLLE' in manifold_args and manifold_args['HLLE'] and 'n_neighbors' in manifold_args:
        HLLE(featrue, labels, imgs, manifold_args['n_neighbors'], **kwargs)
    if 'LTSA' in manifold_args and manifold_args['LTSA'] and 'n_neighbors' in manifold_args:
        LTSA(featrue, labels, imgs, manifold_args['n_neighbors'], **kwargs)
    if 'TSNE' in manifold_args and manifold_args['TSNE']:
        tsne(featrue, labels, imgs, **kwargs)
