#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/15 14:15
@License: (C) Copyright 2013-2023. 
@File: _coor_distance.py
@Desc:

"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def coor_eucl_dist(x):
    """
    calculate pairwise euclidean distance between spots based on coordinates

    Parameters
    ----------
    x: (n_spots,2), numpy 2d array from adata.obsm['spatial'].

    Returns
    -------
    euclidean distance between spots, numpy 2d array with shape of (n_spots,n_spots)
    """

    n_row, n_col = x.shape

    ret_distance = np.empty((n_row, n_row), dtype=np.float32)

    for i in prange(n_row):
        for j in prange(n_row):
            d = 0.0
            for k in prange(n_col):
                d += (x[i, k] - x[j, k]) ** 2
            ret_distance[i, j] = d ** 0.5

    return ret_distance


def coor_gauss_similarity(x, sigma=1):
    """
    calculate Gaussian kernel for similarity (based on coordinate distance).

    s_{i,j}=exp(-(x_i-x_j)^2/2*sigma)

    Parameters
    ----------
    x: (n_spots,2), numpy 2d array from adata.obsm['spatial'].
    sigma: float, band with for controlling gaussian kernel similarity

    Returns
    -------
    Similarity between spots based on distance measure.
    """

    y = coor_eucl_dist(x)
    return np.exp(-(y ** 2 / (2 * sigma)))
