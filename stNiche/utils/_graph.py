#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/15 16:38
@License: (C) Copyright 2013-2023. 
@File: _graph.py
@Desc:

"""
import numpy as np
from numba import njit, prange
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


def crt_graph(x, mode='knn', min_radius=1.0, top_k=12, dist_measure='minkowski'):
    """
    create graph adjacent matrix using coordinates

    Parameters
    ----------
    x: (n_spots,2), numpy 2d array from adata.obsm['spatial'].
    mode: string, any of ['knn', 'radius', 'knn_graph']
    min_radius: minimum radius for neighbor searching based on radius constraint
    top_k: minimum k value for neighbor searching based on the top k nearest neighbors.
    dist_measure: default is `minkowski` (ie., standard euclidean distance),
        others see `https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html`

    Returns
    -------
        adjacent matrix with shape of (n_spots, n_spots), element is eithor 0 or 1
        distance matrix with shape of (n_spots, n_spots) corresponding to the nearest neighbors. Element of -1 value
        means the distance is inf.
    """

    # top_k+1: because NearestNeighbors include self node
    neigh = NearestNeighbors(n_neighbors=top_k+1, radius=min_radius, metric=dist_measure, n_jobs=-1).fit(x)

    n_rows = x.shape[0]
    if mode == 'knn':
        neigh_dist, neigh_ind = neigh.kneighbors(x, return_distance=True)
        return _adj(n_rows, neigh_dist, neigh_ind)

    if mode == 'radius':
        neigh_dist, neigh_ind = neigh.radius_neighbors(x, return_distance=True)
        return _adj(n_rows, neigh_dist, neigh_ind)

    if mode == 'knn_graph':
        adj = kneighbors_graph(x, n_neighbors=top_k+1, mode='distance',
                               metric=dist_measure, include_self=False, n_jobs=-1)  #
        return adj, None


@njit(parallel=True)
def _adj(n_spots, neigh_dist, neigh_ind):

    adj = np.zeros((n_spots, n_spots), dtype=np.float32)
    dist_mat = -1.0 * np.ones((n_spots, n_spots), dtype=np.float32)

    for i in prange(n_spots):
        i_neigh_ind = neigh_ind[i, :]
        for j in prange(len(i_neigh_ind)):
            adj[i, i_neigh_ind[j]] = 1
            dist_mat[i, i_neigh_ind[j]] = neigh_dist[i, j]

    return adj, dist_mat



