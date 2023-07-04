#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/16 12:29
@License: (C) Copyright 2013-2023. 
@File: _filter.py
@Desc:

 adata preprocess, following standard scanpy pipeline
 References: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html

"""

import scanpy as sc


def adata_prep(adata,
               min_genes=200,
               min_cells=3,
               max_genes_by_cnt=2500,
               max_pct_mt=5,
               norm=True,
               target_sum=None,
               log1=True,
               n_hvg=None,
               regress_out=True,
               scale=True):
    """
    preprocess adata following standard scanpy pipeline.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_genes
        Minimum number of genes, for filter cells or spots.
    min_cells
        Minimum number of cells, for filter genes.
    max_genes_by_cnt
        Maximum number of genes a cell or spot has. If `None`, then not performing filtering.
    max_pct_mt
        Minimum percent of mitochondrial genes.
    norm
        Whether perform gene expression normalization.
    target_sum
        gene expression normalization factor. If `None`, then use the value of library size.
    log1
        whether perform log transformation.
    n_hvg
        The number of HVGs. if `None`, then not  not performing HVG calculation.
    regress_out
        Whether regressing out effects of total counts per cell and the percentage of
        mitochondrial genes expressed. Scale the data to unit variance.
    scale
        Whether performing standard normalization on each gene across all cells or spots.

    Returns
    -------
    adata after preprocess.
    """

    # make gene name unique
    adata.var_names_make_unique()

    # basic filtering:
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]

    if max_genes_by_cnt:
        adata = adata[adata.obs.n_genes_by_counts < max_genes_by_cnt, :]

    if norm:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1:
        sc.pp.log1p(adata)

    if n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        adata = adata[:, adata.var.highly_variable]

    if regress_out:
        # Regress out effects of total counts per cell and the percentage of
        # mitochondrial genes expressed. Scale the data to unit variance.
        sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

    if scale:
        # Scale each gene to unit variance. Clip values exceeding standard deviation 10.
        sc.pp.scale(adata, max_value=10)

    return adata




