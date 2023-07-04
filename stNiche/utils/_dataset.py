#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/26 4:01
@License: (C) Copyright 2013-2023. 
@File: _dataset.py
@Desc:

"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    """
    pytorch Dataset
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx, :]).float(), torch.from_numpy(np.array(idx))

