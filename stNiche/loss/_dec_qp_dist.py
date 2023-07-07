#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/30 12:03
@License: (C) Copyright 2013-2023. 
@File: _dec_qp_dist.py
@Desc:

"""

import torch


def cal_dec_q(z, cluster, v):
    """
    ref: DEC: https://arxiv.org/pdf/1511.06335.pdf

    Parameters
    ----------
    z
    cluster
    v

    Returns
    -------

    """
    q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster, 2), 2) / v) + 1e-8)
    q = q.pow((v + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()

    return q


def target_dist(q):
    """
    calculate p distributions

    ref : DEC https://arxiv.org/pdf/1511.06335.pdf
    Parameters
    ----------
    q

    Returns
    -------

    """
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
