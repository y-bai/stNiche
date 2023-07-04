#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/16 10:18
@License: (C) Copyright 2013-2023. 
@File: _normalize.py
@Desc:

"""
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    StandardScaler,
    QuantileTransformer
)


def scaler(x, scale_method='minmax', seed=42, n_quantiles=1000, max_value=10):
    """
    normalize data.

    Parameters
    ----------
    x: array-like of shape (n_spots, n_features)

    scale_method: scale methods, including:
        `minmax`: scale each feature in `x` into range [0,1]. This value is calculated across samples.
        `std_norm`: scale each feature in `x` using standard normalization. This value is calculated across samples.
        `l2_norm`: normalize samples individually to unit norm. This value is not calculated across samples.
        `quantile`: transform features using quantiles information. For a given feature, this transformation tends to
            spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is
            therefore a robust preprocessing scheme.
    seed: random seed value
    n_quantiles: the actual number of quantiles used to discretize the cumulative distribution function. Only used for
        `scale_method='quantile'`

    Returns
    -------
    X_new: ndarray array of shape (n_spots, n_features)
    """

    if scale_method == 'minmax':
        x = MinMaxScaler().fit_transform(x)
    if scale_method == 'std_norm':
        x = StandardScaler().fit_transform(x)
        x[x > max_value] = max_value
    if scale_method == 'l2_norm':
        x = Normalizer().fit_transform(x)
    if scale_method == 'quantile':
        x = QuantileTransformer(n_quantiles=n_quantiles, random_state=seed).fit_transform(x)
    return x




