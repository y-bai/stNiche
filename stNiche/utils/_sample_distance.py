#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/7/7 12:29
@License: (C) Copyright 2013-2023. 
@File: _sample_distance.py
@Desc:

"""
import numpy as np
from numba import njit, prange
from scipy import stats


@njit(parallel=True)
def sample_encl_dist(x, y):
    """

    calculate pairwise euclidean distance between spots based on their features


    Parameters
    ----------
    x
        shape: (n_sample1, n_feature)
    y
        shape: (n_sample2, n_feature)

    Returns
    -------
        shape: (n_sample1, n_sample2)

    """

    n_row1, n_col1 = x.shape
    n_row2, n_col2 = y.shape

    assert n_col1 == n_col2, 'dimension not same when calculating distance'

    res_dist = np.empty((n_row1, n_row2), dtype=np.float32)
    for i in prange(n_row1):
        for j in prange(n_row2):
            d = 0.0
            for k in prange(n_col1):
                d += (x[i, k] - y[j, k]) ** 2
            res_dist[i, j] = d ** 0.5
    return res_dist


def sample_spearman_dist(x, y):
    rho, p = stats.spearmanr(x, y)
    return 0.5 * (1 - rho)

