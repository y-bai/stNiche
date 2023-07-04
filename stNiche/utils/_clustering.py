#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/20 13:10
@License: (C) Copyright 2013-2023. 
@File: _clustering.py
@Desc:

"""


import numpy as np
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, SpectralBiclustering
import torch


class Cluster:
    """

    Cluster

    """

    def __init__(self, method='kmeans', n_clusters=10):
        """

        Cluster, here mainly used for initializing clustering for deep embedding clustering.

        Parameters
        ----------
        method
            clustering methods, could be
                `kmeans`
                `spectral`
                `bispectral`

        n_clusters
            the number of clusters
        """

        self.method = method
        self.n_clusters = n_clusters

    def fit_transform(self, x):
        """

        perform clustering

        Parameters
        ----------
        x
            2d np.array with shape (N_sample, N_z)
        Returns
        -------
        y_labels.astype(np.int32)
            cluster labels

        y_centers.astype(np.float32)
            cluster centers

        """

        if self.method == 'kmeans':
            clustering = MiniBatchKMeans(n_clusters=self.n_clusters,
                                         init='k-means++',
                                         max_iter=500,
                                         batch_size=1024 * 5,
                                         random_state=42,
                                         n_init="auto").partial_fit(x)
        if self.method == 'spectral':
            clustering = SpectralClustering(n_clusters=self.n_clusters,
                                            random_state=42,
                                            gamma=1.0,
                                            affinity='rbf',
                                            assign_labels='kmeans',
                                            n_jobs=-1).fit(x)

        if self.method == 'bispectral':
            clustering = SpectralBiclustering(n_clusters=self.n_clusters,
                                              method="log",
                                              mini_batch=True,
                                              init='k-means++',
                                              random_state=42).fit(x)

        if hasattr(clustering, 'labels_'):
            y_labels = clustering.labels_.astype(int)
        elif hasattr(clustering, 'row_labels_'):
            y_labels = clustering.row_labels_.astype(int)
        else:
            y_labels = clustering.predict(x)

        if hasattr(clustering, 'cluster_centers_'):
            y_centers = clustering.cluster_centers_
        else:
            y_centers = []
            uni_labels = np.unique(y_labels).tolist()
            for i in uni_labels:
                y_centers.append(np.means(x[y_labels == i, :], axis=0))
            y_centers = np.array(y_centers)
        return y_labels.astype(np.int32), y_centers.astype(np.float32)

