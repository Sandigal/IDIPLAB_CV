# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:37:39 2018

@author: Sandiagal
"""

import visul
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE

# %%


def plot_embedding(X, labels, imgs, title=None):
    # Scale and visualize the embedding vectors
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(16, 9))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(len(X)):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(
                    imgs[i], zoom=0.05, cmap=plt.cm.gray_r),
                X[i], pad=.0)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def Random(X, labels, imgs):
    # Random 2D projection using a random unitary matrix
    print("Computing random projection")
    t = time()
    rp = SparseRandomProjection(
        n_components=2, random_state=0)
    X_projected = rp.fit_transform(X)
    plot_embedding(X_projected, labels, imgs, "Random Projection of the dataset (time %.2fs)" %
                   (time() - t))


def RandomTrees(X, labels, imgs):
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
                   (time() - t))


def mds(X, labels, imgs):
    # MDS  embedding of the dataset dataset
    print("Computing MDS embedding")
    clf = MDS(n_components=2, n_init=1, max_iter=100)
    t = time()
    X_mds = clf.fit_transform(X)
    print("Done. Stress: %f" % clf.stress_)
    plot_embedding(X_mds, labels, imgs,
                   "MDS embedding of the dataset (time %.2fs)" %
                   (time() - t))


def PCA(X, labels, imgs):
    # Projection on to the first 2 principal components
    print("Computing Principal Components projection")
    t = time()
    X_pca = TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca, labels, imgs,
                   "Principal Components projection of the dataset (time %.2fs)" %
                   (time() - t))


def LinearDiscriminant(X, labels, imgs):
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
                   (time() - t))


def isomap(X, labels, imgs, n_neighbors):
    # Isomap projection of the dataset dataset
    print("Computing Isomap embedding")
    t = time()
    X_iso = Isomap(n_neighbors, n_components=2).fit_transform(X)
    print("Done.")
    plot_embedding(X_iso, labels, imgs,
                   "Isomap projection of the dataset (time %.2fs)" %
                   (time() - t))


def Spectral(X, labels, imgs):
    # Spectral embedding of the dataset dataset
    print("Computing Spectral embedding")
    embedder = SpectralEmbedding(n_components=2, random_state=0,
                                 eigen_solver="arpack")
    t = time()
    X_se = embedder.fit_transform(X)

    plot_embedding(X_se, labels, imgs,
                   "Spectral embedding of the dataset (time %.2fs)" %
                   (time() - t))


def LLE(X, labels, imgs, n_neighbors):
    # Locally linear embedding of the dataset dataset
    print("Computing Locally Linear embedding")
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                 method='standard')
    t = time()
    X_lle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_lle, labels, imgs,
                   "Locally Linear Embedding of the dataset (time %.2fs)" %
                   (time() - t))


def ModifiedLLE(X, labels, imgs, n_neighbors):
    # Modified Locally linear embedding of the dataset dataset
    print("Computing modified LLE embedding")
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                 method='modified')
    t = time()
    X_mlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_mlle, labels, imgs,
                   "Modified Locally Linear Embedding of the dataset (time %.2fs)" %
                   (time() - t))


def HLLE(X, labels, imgs, n_neighbors):
    # Hessian Locally Linear embedding of the dataset dataset
    print("Computing Hessian LLE embedding")
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                 method='hessian')
    t = time()
    X_hlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_hlle, labels, imgs,
                   "Hessian Locally Linear Embedding of the dataset (time %.2fs)" %
                   (time() - t))


def LTSA(X, labels, imgs, n_neighbors):
    # LTSA embedding of the dataset dataset
    print("Computing LTSA embedding")
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2,
                                 method='ltsa')
    t = time()
    X_ltsa = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_ltsa, labels, imgs,
                   "LTSA of the dataset (time %.2fs)" %
                   (time() - t))


def tsne(X, labels, imgs):
    # t-SNE embedding of the dataset dataset
    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne, labels, imgs,
                   "t-SNE embedding of the dataset (time %.2fs)" %
                   (time() - t))


def manifold(imgs, labels, manifold_args):
    imgs = np.copy(imgs)
    visul.overall(imgs, 64)

    X = imgs.reshape((len(imgs), -1))
    X = X/255

    if 'Random' in manifold_args and manifold_args['Random']:
        Random(X, labels, imgs)
    if 'RandomTrees' in manifold_args and manifold_args['RandomTrees']:
        RandomTrees(X, labels, imgs)
    if 'MDS' in manifold_args and manifold_args['MDS']:
        mds(X, labels, imgs)
    if 'PCA' in manifold_args and manifold_args['PCA']:
        PCA(X, labels, imgs)
    if 'LinearDiscriminant' in manifold_args and manifold_args['LinearDiscriminant']:
        LinearDiscriminant(X, labels, imgs)
    if 'Isomap' in manifold_args and manifold_args['Isomap'] and 'n_neighbors' in manifold_args:
        isomap(X, labels, imgs, manifold_args['n_neighbors'])
    if 'Spectral' in manifold_args and manifold_args['Spectral']:
        Spectral(X, labels, imgs)
    if 'LLE' in manifold_args and manifold_args['LLE'] and 'n_neighbors' in manifold_args:
        LLE(X, labels, imgs, manifold_args['n_neighbors'])
    if 'ModifiedLLE' in manifold_args and manifold_args['ModifiedLLE'] and 'n_neighbors' in manifold_args:
        ModifiedLLE(X, labels, imgs, manifold_args['n_neighbors'])
    if 'HLLE' in manifold_args and manifold_args['HLLE'] and 'n_neighbors' in manifold_args:
        HLLE(X, labels, imgs, manifold_args['n_neighbors'])
    if 'LTSA' in manifold_args and manifold_args['LTSA'] and 'n_neighbors' in manifold_args:
        LTSA(X, labels, imgs, manifold_args['n_neighbors'])
    if 'TSNE' in manifold_args and manifold_args['TSNE']:
        tsne(X, labels, imgs)
