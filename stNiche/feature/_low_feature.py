#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/16 9:05
@License: (C) Copyright 2013-2023. 
@File: _low_feature.py
@Desc:

feature reduction using classic methods, which we call as low feature reduction, compared to feature embedding using
deep learning approaches such as VAE
# https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py

"""

from sklearn.decomposition import KernelPCA, PCA, DictionaryLearning, NMF
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.manifold import Isomap, MDS, TSNE


def low_embedding(x, *, n_comp=20,
                  embedding='pca',
                  kpca_kernel='rbf', kpca_gamma=2,
                  dictl_trans_alpha=0.1,
                  isomap_n_neigh=8,
                  tsne_perplexity=30.0,
                  seed=42):
    """
    feature reduction using classic methods

    Parameters
    ----------
    x: array-like of shape (n_spots, n_features).
    n_comp: number of output features.
    embedding: method of feature reduction.
        'pca': Principal component analysis;
        'kpca': Non-linear dimensionality reduction using kernels and PCA;
        'dictl': Dictionary learning.
        'nmf': Non-Negative Matrix Factorization (NMF).
        'nca': Neighborhood Components Analysis.
        'isomap': Manifold learning based on Isometric Mapping.
        'tsne': T-distributed Stochastic Neighbor Embedding.
        'mds': Multidimensional scaling.
    kpca_kernel: Kernel used for PCA.
    kpca_gamma: Kernel coefficient for rbf, poly and sigmoid kernels.
    dictl_trans_alpha: for dictionary learning. If algorithm='lasso_lars' or algorithm='lasso_cd',
        alpha is the penalty applied to the L1 norm. If algorithm='threshold', alpha is the absolute value
        of the threshold below which coefficients will be squashed to zero.
    isomap_n_neigh: for IsoMAP, Number of neighbors to consider for each point.
    tsne_perplexity: for TSNE, The perplexity is related to the number of nearest neighbors that is used in other
        manifold learning algorithms. Larger datasets usually require a larger perplexity.
        Consider selecting a value between 5 and 50.
    seed: random seed value

    Returns
    -------
    X_new: ndarray array of shape (n_spots, n_comp)
    """

    emb_trans = None
    # feature reduction
    if embedding == 'kpca':
        emb_trans = KernelPCA(n_components=n_comp, kernel=kpca_kernel, gamma=kpca_gamma, random_state=seed, n_jobs=-1)
    if embedding == 'pca':
        emb_trans = PCA(n_components=n_comp, random_state=seed)
    if embedding == 'dictl':
        emb_trans = DictionaryLearning(n_components=n_comp, transform_algorithm='lasso_lars',
                                       transform_alpha=dictl_trans_alpha, random_state=seed, n_jobs=-1)
    if embedding == 'nmf':
        emb_trans = NMF(n_components=n_comp, random_state=seed, max_iter=1000,
                        alpha_W=0.000005, alpha_H='same', l1_ratio=0.5)
    if embedding == 'nca':
        emb_trans = NeighborhoodComponentsAnalysis(n_components=n_comp, init="pca", random_state=seed)
    if embedding == 'isomap':
        emb_trans = Isomap(n_neighbors=isomap_n_neigh, n_components=n_comp, n_jobs=-1, metric='minkowski', p=2)
    if embedding == 'tsne':
        emb_trans = TSNE(n_components=n_comp, perplexity=tsne_perplexity, metric='euclidean',
                         n_jobs=-1, random_state=seed)
    if embedding == 'mds':
        emb_trans = MDS(n_components=n_comp, n_init=1, n_jobs=-1, random_state=seed, normalized_stress="auto")

    assert emb_trans is not None, 'feature reduction failed'
    return emb_trans.fit_transform(x)
