#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/15 11:13
@License: (C) Copyright 2013-2023. 
@File: main.py
@Desc:

"""

# import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import random

import torch
import torch.nn as nn
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

import stNiche as stn
from captum.attr import Saliency, IntegratedGradients, InputXGradient, DeepLift, FeatureAblation, FeaturePermutation

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager  # to solve: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.
import seaborn as sns


def config_rc(dpi=400, font_size=5, lw=1.):
    # matplotlib.rcParams.keys()
    rc = {
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.dpi': dpi, 'axes.linewidth': lw,
    }  # 'figure.figsize':(11.7/1.5,8.27/1.5)

    sns.set(style='ticks', rc=rc)
    sns.set_context("paper")

    mpl.rcParams.update(rc)

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    # mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['axes.unicode_minus'] = False  # negative minus sign

    sc.settings.set_figure_params(vector_friendly=True)
    sc.set_figure_params(facecolor="white", figsize=(8, 8))
    sc.settings.verbosity = 3


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def crt_data2(f_name, n_comps=20, min_genes=100, min_cells=3, max_genes_by_cnt=2500, regress_out=True):
    full_name = './data/' + f_name + '.h5ad'
    adata = sc.read_h5ad(full_name)

    adata.obs['pred_celltype'] = adata.obsm['cell_frac'].idxmax(axis=1)

    # spatial: first column: array_col(width), second column: array_row(height)
    # origial adata from H.K:
    # adata.obs['x'] = df_meta['row'].values
    # adata.obs['y'] = df_meta['col'].values
    adata.obsm['spatial'] = adata.obs[['y', 'x']].values

    # adata filter and preprocess
    adata = stn.pp.adata_prep(adata,
                              min_genes=min_genes, min_cells=min_cells,
                              max_genes_by_cnt=max_genes_by_cnt, max_pct_mt=5, norm=True,
                              target_sum=1e4, log1=True, n_hvg=2500, regress_out=regress_out, scale=False)
    print('After preprocess:\n', adata)

    # adata dimension reduction
    sc.tl.pca(adata, n_comps=n_comps, random_state=42)

    # save preprocessed data
    adata.write_h5ad('./results/' + f_name + '_prep.h5ad')

    x = np.concatenate((adata.obsm['X_pca'], adata.obsm['cell_frac'].to_numpy()), axis=1)

    # x = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    # adata.obsm['X_kpca'] = stn.feature.low_embedding(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
    #                                                    n_comp=n_comps,
    #                                                    embedding='kpca')
    # x = np.concatenate((adata.obsm['X_kpca'], adata.obsm['cell_frac'].to_numpy()), axis=1)

    # normalize
    x = stn.pp.scaler(x, scale_method='std_norm', max_value=10)

    # adj
    # spatial: first column: array_col(width), second column: array_row(height)
    spatial = adata.obsm['spatial']
    adj, _ = stn.utils.crt_graph(spatial, mode='knn_graph', top_k=8)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    # torch.geometric dataset
    g_data = Data(x=torch.tensor(x, dtype=torch.float32),
                  edge_index=edge_index)

    torch.save(g_data, r'./results/' + f_name + '_graph_data.pt')


def crt_data5(f_name):

    full_name = './data/' + f_name + '.h5ad'
    adata = sc.read_h5ad(full_name)

    # lr_effect_pca = stn.feature.low_embedding(adata.obsm['lr_effect'], n_comp=40)  # 40
    # pathway_activity = stn.feature.low_embedding(adata.obsm['pathway_activity'], n_comp=20)  #30

    x = np.concatenate((adata.obsm['gene_effect_filter'].to_numpy(),
                        adata.obsm['cell_frac'].to_numpy(),
                        adata.obsm['pathway_activity_filter'].to_numpy(),
                        adata.obsm['lr_effect_filter'].to_numpy()), axis=1)

    # normalize
    x = stn.pp.scaler(x, scale_method='std_norm', max_value=10)

    # adj
    # spatial: first column: array_col(width), second column: array_row(height)
    spatial = adata.obsm['spatial']

    # spatial clustering region will be large if top_k is large
    adj, _ = stn.utils.crt_graph(spatial, mode='knn_graph', top_k=8)

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    # torch.geometric dataset
    g_data = Data(x=torch.tensor(x, dtype=torch.float32),
                  edge_index=edge_index)

    torch.save(g_data, r'./results/' + f_name + '_graph_data.pt')


def crt_data4(f_name, min_genes=100, min_cells=3, max_genes_by_cnt=2500, regress_out=True):

    from scipy.sparse import issparse

    full_name = './data/' + f_name + '.h5ad'
    adata = sc.read_h5ad(full_name)

    adata.obs['pred_celltype'] = adata.obsm['cell_frac'].idxmax(axis=1)

    # spatial: first column: array_col(width), second column: array_row(height)
    # origial adata from H.K:
    # adata.obs['x'] = df_meta['row'].values
    # adata.obs['y'] = df_meta['col'].values
    adata.obsm['spatial'] = adata.obs[['y', 'x']].values

    # adata filter and preprocess
    adata = stn.pp.adata_prep(adata,
                              min_genes=min_genes, min_cells=min_cells,
                              max_genes_by_cnt=max_genes_by_cnt, max_pct_mt=5, norm=True,
                              target_sum=1e4, log1=True, n_hvg=500, regress_out=regress_out, scale=False)
    print('After preprocess:\n', adata)

    # adata dimension reduction
    # sc.tl.pca(adata, n_comps=n_comps, random_state=42)

    # save preprocessed data
    adata.write_h5ad('./results/' + f_name + '_prep.h5ad')

    # lr_effect_pca = stn.feature.low_embedding(adata.obsm['lr_effect'], n_comp=40)  # 40
    # pathway_activity = stn.feature.low_embedding(adata.obsm['pathway_activity'], n_comp=20)  #30

    x = np.concatenate((adata.X.A if issparse(adata.X) else adata.X, adata.obsm['cell_frac'].to_numpy(),
                        adata.obsm['pathway_activity'].to_numpy(), adata.obsm['lr_effect'].to_numpy()), axis=1)

    # x = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    # adata.obsm['X_kpca'] = stn.feature.low_embedding(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
    #                                                    n_comp=n_comps,
    #                                                    embedding='kpca')
    # x = np.concatenate((adata.obsm['X_kpca'], adata.obsm['cell_frac'].to_numpy()), axis=1)

    # normalize
    x = stn.pp.scaler(x, scale_method='std_norm', max_value=10)

    # adj
    # spatial: first column: array_col(width), second column: array_row(height)
    spatial = adata.obsm['spatial']

    # spatial clustering region will be large if top_k is large
    adj, _ = stn.utils.crt_graph(spatial, mode='knn_graph', top_k=8)

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    # torch.geometric dataset
    g_data = Data(x=torch.tensor(x, dtype=torch.float32),
                  edge_index=edge_index)

    torch.save(g_data, r'./results/' + f_name + '_graph_data.pt')


def crt_data3(f_name, n_comps=20, min_genes=100, min_cells=3, max_genes_by_cnt=2500, regress_out=True):
    full_name = './data/' + f_name + '.h5ad'
    adata = sc.read_h5ad(full_name)

    adata.obs['pred_celltype'] = adata.obsm['cell_frac'].idxmax(axis=1)

    # spatial: first column: array_col(width), second column: array_row(height)
    # origial adata from H.K:
    # adata.obs['x'] = df_meta['row'].values
    # adata.obs['y'] = df_meta['col'].values
    adata.obsm['spatial'] = adata.obs[['y', 'x']].values

    # adata filter and preprocess
    adata = stn.pp.adata_prep(adata,
                              min_genes=min_genes, min_cells=min_cells,
                              max_genes_by_cnt=max_genes_by_cnt, max_pct_mt=5, norm=True,
                              target_sum=1e4, log1=True, n_hvg=2500, regress_out=regress_out, scale=False)
    print('After preprocess:\n', adata)

    # adata dimension reduction
    sc.tl.pca(adata, n_comps=n_comps, random_state=42)

    # save preprocessed data
    adata.write_h5ad('./results/' + f_name + '_prep.h5ad')

    lr_effect_pca = stn.feature.low_embedding(adata.obsm['lr_effect'], n_comp=40)  # 40
    pathway_activity = stn.feature.low_embedding(adata.obsm['pathway_activity'], n_comp=20)  #30

    # x = np.concatenate((adata.obsm['X_pca'], adata.obsm['cell_frac'].to_numpy(),
    #                     adata.obsm['pathway_activity'].to_numpy(), lr_effect_pca), axis=1)

    x = np.concatenate((adata.obsm['X_pca'], adata.obsm['cell_frac'].to_numpy(),
                        pathway_activity, lr_effect_pca), axis=1)

    # x = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    # adata.obsm['X_kpca'] = stn.feature.low_embedding(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
    #                                                    n_comp=n_comps,
    #                                                    embedding='kpca')
    # x = np.concatenate((adata.obsm['X_kpca'], adata.obsm['cell_frac'].to_numpy()), axis=1)

    # normalize
    x = stn.pp.scaler(x, scale_method='std_norm', max_value=10)

    # adj
    # spatial: first column: array_col(width), second column: array_row(height)
    spatial = adata.obsm['spatial']

    # spatial clustering region will be large if top_k is large
    adj, _ = stn.utils.crt_graph(spatial, mode='knn_graph', top_k=8)

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    # torch.geometric dataset
    g_data = Data(x=torch.tensor(x, dtype=torch.float32),
                  edge_index=edge_index)

    torch.save(g_data, r'./results/' + f_name + '_graph_data.pt')


def crt_data_tmp(n_comps=20, min_genes=100, min_cells=3, max_genes_by_cnt=2500, regress_out=True):
    f_h5ad = r'./data/LC4-B_FE7_web4_bin50_Cluster.h5ad'
    f_csv = r'./data/LC4-B_FE7_web4_bin50_CelltypeTrans_Spotlight.txt'

    # read csv file containing cell type fraction per spot
    spotlight_df = pd.read_csv(f_csv, sep='\t')
    # read h5ad
    adata = sc.read_h5ad(f_h5ad)
    p_series_mt = spotlight_df['percent.mt']
    p_series_mt.index = adata.obs_names
    adata.obs['percent.mt'] = p_series_mt

    p_serise_ct = spotlight_df['predict_CellType'].astype('category')
    p_serise_ct.index = adata.obs_names
    adata.obs['pred_celltype'] = p_serise_ct

    cell_frac = spotlight_df[
        ['B.cell', 'Cholangiocyte', 'DC', 'Endothelial', 'Fibroblast',
         'Hepatocyte', 'Macrophage', 'Malignant', 'NK', 'Plasma', 'T.cell']
    ].copy()
    cell_frac.index = adata.obs_names
    adata.obsm['cell_frac'] = cell_frac

    ###
    pw_genes_df = pd.read_csv('./data/LC4-B_FE7_web4_bin50_Cluster_cancer.cNMF.program.genes.exp.new.tsv',
                              sep='\t').iloc[:, 1:-2]
    lr_df = pd.read_csv('./data/LC4-B_FE7_web4_bin50_Cluster_bin.LR.effect.tsv', sep='\t').iloc[:, 3:]
    pw_genes_df.index = adata.obs_names
    lr_df.index = adata.obs_names
    adata.obsm['pathway_genes'] = pw_genes_df
    adata.obsm['lr_effects'] = lr_df

    # adata filter and preprocess
    adata = stn.pp.adata_prep(adata,
                              min_genes=min_genes, min_cells=min_cells,
                              max_genes_by_cnt=max_genes_by_cnt, max_pct_mt=5, norm=True,
                              target_sum=1e4, log1=True, n_hvg=2500, regress_out=regress_out, scale=False)

    print('After preprocess:\n', adata)

    # adata dimension reduction
    sc.tl.pca(adata, n_comps=n_comps, random_state=42)

    # save preprocessed data
    # adata.write_h5ad(r'./results/LC4-B_FE7_web4_bin50_Cluster_prep.h5ad')

    # add other features
    adata.write_h5ad(r'./results/LC4-B_FE7_web4_bin50_Cluster2_prep.h5ad')

    lr_effect_pca = stn.feature.low_embedding(adata.obsm['lr_effects'])

    x = np.concatenate((adata.obsm['X_pca'], adata.obsm['cell_frac'].to_numpy(),
                        adata.obsm['pathway_genes'].to_numpy(), lr_effect_pca), axis=1)

    # x = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    # adata.obsm['X_kpca'] = stn.feature.low_embedding(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
    #                                                    n_comp=n_comps,
    #                                                    embedding='kpca')
    # x = np.concatenate((adata.obsm['X_kpca'], adata.obsm['cell_frac'].to_numpy()), axis=1)

    # normalize
    x = stn.pp.scaler(x, scale_method='std_norm', max_value=10)

    # adj
    # spatial: first column: array_col(width), second column: array_row(height)
    spatial = adata.obsm['spatial']
    print(spatial.shape)
    adj, _ = stn.utils.crt_graph(spatial, mode='knn_graph', top_k=8)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    print(edge_index)
    # torch.geometric dataset
    g_data = Data(x=torch.tensor(x, dtype=torch.float32),
                  edge_index=edge_index)

    # torch.save(g_data, r'./results/LC4-B_FE7_web4_bin50_Cluster_graph_data.pt')
    # added other features
    torch.save(g_data, r'./results/LC4-B_FE7_web4_bin50_Cluster2_graph_data.pt')

#
# def crt_data():
#     f_h5ad = r'./data/LC4-B_FE7_web4_bin50_Cluster.h5ad'
#     f_csv = r'./data/LC4-B_FE7_web4_bin50_CelltypeTrans_Spotlight.txt'
#
#     # read csv file containing cell type fraction per spot
#     spotlight_df = pd.read_csv(f_csv, sep='\t')
#     # read h5ad
#     adata = sc.read_h5ad(f_h5ad)
#     p_series_mt = spotlight_df['percent.mt']
#     p_series_mt.index = adata.obs_names
#     adata.obs['percent.mt'] = p_series_mt
#
#     p_serise_ct = spotlight_df['predict_CellType'].astype('category')
#     p_serise_ct.index = adata.obs_names
#     adata.obs['pred_celltype'] = p_serise_ct
#
#     cell_frac = spotlight_df[
#         ['B.cell', 'Cholangiocyte', 'DC', 'Endothelial', 'Fibroblast',
#          'Hepatocyte', 'Macrophage', 'Malignant', 'NK', 'Plasma', 'T.cell']
#     ].copy()
#     cell_frac.index = adata.obs_names
#     adata.obsm['cell_frac'] = cell_frac
#
#     # adata filter and preprocess
#     adata = stn.pp.adata_prep(adata,
#                               min_genes=100, min_cells=3, max_genes_by_cnt=2500, max_pct_mt=5, norm=True,
#                               target_sum=1e4, log1=True, n_hvg=2500, regress_out=True, scale=False)
#     print('After preprocess:\n', adata)
#
#     # adata dimension reduction
#     n_comps = 20
#     sc.tl.pca(adata, n_comps=n_comps, random_state=42)
#
#     # save preprocessed data
#     adata.write_h5ad(r'./results/prep_adata.pt')
#
#     x = np.concatenate((adata.obsm['X_pca'], adata.obsm['cell_frac'].to_numpy()), axis=1)
#
#     # x = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
#     # adata.obsm['X_kpca'] = stn.feature.low_embedding(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
#     #                                                    n_comp=n_comps,
#     #                                                    embedding='kpca')
#     # x = np.concatenate((adata.obsm['X_kpca'], adata.obsm['cell_frac'].to_numpy()), axis=1)
#
#     # normalize
#     x = stn.pp.scaler(x, scale_method='std_norm', max_value=10)
#
#     # adj
#     # spatial: first column: array_col(width), second column: array_row(height)
#     spatial = adata.obsm['spatial']
#     print(spatial.shape)
#     adj, _ = stn.utils.crt_graph(spatial, mode='knn_graph', top_k=8)
#     edge_index, edge_weight = from_scipy_sparse_matrix(adj)
#     print(edge_index)
#     # torch.geometric dataset
#     g_data = Data(x=torch.tensor(x, dtype=torch.float32),
#                   edge_index=edge_index)
#
#     torch.save(g_data, r'./results/graph_data_knn24.pt')


def plot_out(adata, y_pred, z, out_fname, spot_size=1.5):
    """

    Parameters
    ----------
    adata
    y_pred
    z
    out_fname
    spot_size: should be 80 if spatial id too large

    Returns
    -------

    """
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), subplot_kw={'aspect': 'equal'})
    axs = axs.ravel()

    spot_size = spot_size
    adata.obs['y_pred'] = y_pred.astype('str')
    sc.pl.spatial(adata, color="y_pred", spot_size=spot_size, frameon=False,
                  show=False, ax=axs[0])

    adata.obsm['niche_emb'] = z
    sc.pp.neighbors(adata, use_rep='niche_emb', key_added='niche_emb_neighbor')
    sc.tl.umap(adata, neighbors_key='niche_emb_neighbor')
    sc.pl.umap(adata, color=["y_pred"], wspace=0.4, neighbors_key='niche_emb_neighbor', show=False, ax=axs[1])

    plt.savefig(out_fname, bbox_inches='tight', format='pdf', dpi=400)


def main(f_name):

    # load data
    # g_data = torch.load(r'./results/graph_data_knn24.pt')
    # adata = sc.read_h5ad(r'./results/prep_adata.pt')

    g_data = torch.load('./results/' + f_name + '_graph_data.pt')
    # adata = sc.read_h5ad('./results/' + f_name + '_prep.h5ad')
    # 2023-10-10
    adata = sc.read_h5ad('./data/' + f_name + '.h5ad')

    x = g_data.x.numpy()

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    concat = True
    # 2023-10-10: in_features:328
    in_features = x.shape[-1]
    print(in_features)

    n_hiddens = 1
    # 2023-08
    # n_head = 4

    # 2023-10-10
    n_head = 2
    n_clusters = adata.obsm['cell_frac'].shape[-1] + 2
    print('n_clusters: ', n_clusters)
    # increase the number of neurons on the last layers could increase the clusters
    # dim_hidden = [512, 256, 512, 256, 256, 256, 64] (2-head)

    # 2023-08
    # dim_hidden = [256, 128, 256, 128, 128, 128, 32]

    # 2023-10-10
    # dim_hidden = [128, 128, 64, 32, 32]  # opt1
    # dim_hidden = [128, 128, 64, 64, 32]  # opt2
    # dim_hidden = [128, 128, 32, 32, 32]  # opt3
    dim_hidden = [128, 64, 64, 128]

    n_z = 64   # n_z = 64, (2023-08)

    #
    learning_rate = 0.01  # 0.01 (2023-08)

    max_epochs = 50
    l2_weight = 0.00001  # 0.000001 (2023-08) more smaller, less  clusters
    # 2023-10-10 opt: l2_weight = 0.00001
    batch_size = 512  # 1024  * 2 (2023-08)

    alpha = 1  # emb kl qp

    # # tau optimal values
    # alpha_emb_rescon = 0.1
    # alpha_emb_cluster = 0.01
    # alpha_gat_cluster = 5.0  # gat kl qp

    # # nn.Tanh suboptimal values
    # alpha_emb_rescon = 0.01
    # alpha_emb_cluster = 0.001
    # alpha_gat_cluster = 5.0  # gat kl qp

    # nn.Tanh optimal values
    alpha_emb_rescon = 0.01   # 0.1
    alpha_emb_cluster = 0.001   # 0.01
    alpha_gat_cluster = 15.0  # gat kl qp

    beta = 0.001
    gamma = 0.0001  # https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py#L148
    c_max = 25  # beta-VAE: https://arxiv.org/pdf/1804.03599.pdf
    lam = 1e-5  # https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py

    # embedding = 'vae'
    # model_type = 'vanilla_vae'  # beta = 0.001
    #
    # embedding = 'vae'
    # model_type = 'beta_vae'  # gamma = 0.0001
    #
    # embedding = 'ae'
    # model_type = 'vanilla_ae'

    embedding = 'ae'
    model_type = 'contractive_ae'

    combine = 'gat_emb'   # feature combination

    embedding_model_path = r'./results/{0}_pretrain_emb_net_{1}.ckpt'.format(f_name, model_type)

    #################################################################################################
    # embedding pretrain
    if embedding == 'vae':
        emb_model = stn.embed.VAE(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z, n_head=n_head, concat=concat)
    if embedding == 'ae':
        emb_model = stn.embed.AE(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z, n_head=n_head, concat=concat)
    print(emb_model)
    print(emb_model.state_dict().keys())
    # print(emb_model.z_layer[0].weight)
    emb_pretrainer = stn.train.EmbNetTrainer(
        emb_model, beta=beta, gamma=gamma, c_max=c_max, embedding=embedding, model_type=model_type,
        batch_size=batch_size, device=device, learning_rate=learning_rate, lam=lam,
        max_epochs=max_epochs, l2_weight=l2_weight)
    emb_pretrainer.fit(x)
    init_z = emb_pretrainer.predict(x)
    emb_pretrainer.save_model(embedding_model_path)

    print(emb_model)
    print(init_z.shape)

    # init clustering
    y_labels, y_centers = stn.utils.Cluster(n_clusters=n_clusters).fit_transform(init_z)
    print(y_centers.shape)
    # plot_out(adata, y_labels, init_z, r'./results/{0}_ini_pred_{1}.pdf'.format(model_type, combine))

    #################################################################################################
    # # GRAPH NET
    # graph_cluster = stn.embed.GraphNetCluster(in_features, dim_hidden, n_clusters,
    #                                           n_hiddens=n_hiddens, n_z=n_z, n_head=n_head, concat=concat)
    # graph_trainer = stn.train.GraphNetClusterTrainer(graph_cluster, y_centers, y_labels, tol=0.000000001,
    #                                                device=device, learning_rate=learning_rate,
    #                                                max_epochs=50, l2_weight=l2_weight)
    # graph_trainer.fit(g_data)
    # y_pred_q, gat_z = graph_trainer.predict(g_data)
    # np.savez(r'./results/gat_predict.npz', y_pred_label_q=y_pred_q, final_z=gat_z)
    #
    # # f = np.load(r'./results/gat_predict.npz')
    # plot_out(adata, y_pred_q, gat_z,
    #          r'./results/{0}_final_pred_{1}.pdf'.format(model_type, combine))

    ##################################################################################################
    # EMB_NET
    # emb_cluster = stn.embed.EmbNetCluster(in_features, dim_hidden, n_clusters, n_hiddens=n_hiddens,
    #                                       n_z=n_z, n_head=n_head, concat=concat,
    #                                       embedding=embedding, pretrain_path=embedding_model_path)
    # # Trainer
    # emb_trianer = stn.train.EmbNetClusterTrainer(emb_cluster, y_centers, y_labels, tol=0.001, batch_size=batch_size,
    #                                              alpha=alpha, beta=beta, gamma=gamma, c_max=c_max, lam=lam,
    #                                              device=device, embedding=embedding, model_type=model_type,
    #                                              learning_rate=learning_rate, max_epochs=50, l2_weight=l2_weight)
    #
    # emb_trianer.fit(x)
    # z, y_pred = emb_trianer.predict(x)
    # np.savez(r'./results/emb_predict.npz', y_pred_label_q=y_pred, final_z=z)
    #
    # plot_out(adata, y_pred, z, r'./results/{0}_final_pred_{1}.pdf'.format(model_type, combine))

    #################################################################################################
    # Graph Embedding Net

    graph_emb_model = stn.tool.GraphEmbNetCluster(
        in_features, dim_hidden, n_clusters, n_hiddens=n_hiddens, n_z=n_z,
        n_head=n_head, concat=concat, tau=0.001, gate=True,
        embedding=embedding, pretrain_path=embedding_model_path)
    print(graph_emb_model)

    graph_emb_trainer = stn.train.GraphEmbNetClusterTrainer(
        graph_emb_model, y_centers, y_labels,
        tol=1e-6, embedding=embedding, model_type=model_type,
        alpha_emb_recons=alpha_emb_rescon, alpha_emb_cluster=alpha_emb_cluster, alpha_gat_cluster=alpha_gat_cluster,
        beta=beta, gamma=gamma, c_max=c_max, lam=lam,
        device=device, learning_rate=learning_rate, max_epochs=max_epochs, l2_weight=l2_weight)

    graph_emb_trainer.fit(g_data)
    y_pred_q, y_pred_gat, gat_h, emb_z = graph_emb_trainer.predict(g_data)
    final_model_path = r'./results/{0}_final_graph_emb_net_{1}.ckpt'.format(f_name, graph_emb_trainer.embedding)
    graph_emb_trainer.save_model(final_model_path)

    np.savez(r'./results/{0}_predict.npz'.format(f_name), y_pred_label_q=y_pred_q,
             y_pred_gat=y_pred_gat, gat_h=gat_h, emb_z=emb_z)
    # f = np.load(r'./results/predict.npz')

    spot_size = 1.5
    if np.max(adata.obsm['spatial']) > 10000:
        spot_size = 80.0
    if np.max(adata.obsm['spatial']) < 500:
        spot_size = 1.2

    plot_out(adata, y_pred_gat, gat_h, r'./results/{0}_{1}_final_pred_{2}.pdf'.format(f_name, model_type, combine),
             spot_size=spot_size)
    # plot_out(adata, y_pred_q, emb_z, r'./results/{0}_final_pred2_{1}.pdf'.format(model_type, combine))
    # plot_out(adata, y_pred_gat, gat_h, r'./results/{0}_final_pred3_{1}.pdf'.format(model_type, combine))


def explain(method, f_name, target=[0], device='cpu'):

    g_data = torch.load('./results/' + f_name + '_graph_data.pt')
    # adata = sc.read_h5ad('./results/' + f_name + '_prep.h5ad')
    # 2023-10-10
    adata = sc.read_h5ad('./data/' + f_name + '.h5ad')

    x = g_data.x.numpy()
    print(x.shape)

    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    # # device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # print(device)

    concat = True
    in_features = x.shape[-1]
    n_hiddens = 1

    # 2023-08
    # n_head = 4

    # 2023-10-10
    n_head = 2

    n_clusters = adata.obsm['cell_frac'].shape[-1] + 2  # original 12, increase this with lr
    # increase the number of neurons on the last layers could increase the clusters
    # dim_hidden = [512, 256, 512, 256, 256, 256, 64] (2-head)
    #
    # 2023-08
    # dim_hidden = [256, 128, 256, 128, 128, 128, 32]  # [256, 512, 256, 64] (PCA)

    dim_hidden = [128, 64, 64, 128]
    n_z = 64
    embedding = 'ae'

    model_path = r'./results/{0}_final_graph_emb_net_{1}.ckpt'.format(f_name, embedding)

    model = stn.tool.GraphEmbNetCluster(
        in_features, dim_hidden, n_clusters, n_hiddens=n_hiddens, n_z=n_z,
        n_head=n_head, concat=concat, tau=0.001, gate=True,
        embedding=embedding, pretrain_path=None)

    model.load_state_dict(torch.load(model_path))
    my_model = model.graph_net

    my_model = my_model.to(device)
    # print(my_model)

    g_data = g_data.to(device)

    class WModel(nn.Module):

        def __init__(self, x_model):
            super(WModel, self).__init__()
            self.model = x_model

        def forward(self, x_feat, adj):
            gat_h, emb_recons, emb_z = self.model(x_feat[0], adj[0])

            return gat_h

    mm = WModel(my_model)
    if method == 'ixg':
        ig = InputXGradient(mm)  # InputXGradient
    if method == 'ig':
        ig = IntegratedGradients(mm)
    if method == 'fa':
        ig = FeatureAblation(mm)
    if method == 'fp':
        ig = FeaturePermutation(mm)

    res_dict = {}
    for i_target in target:
        res = ig.attribute(g_data.x.unsqueeze(0),
                           target=i_target,
                           additional_forward_args=(g_data.edge_index.unsqueeze(0),))
        if device == 'cpu':
            res_dict[i_target] = res.detach().numpy()[0]
        else:
            res_dict[i_target] = res.detach().cpu().numpy()[0]
    return res_dict


def plot_niche(f_name, grp):
    adata_f = r'./results/{0}_prep.h5ad'.format(f_name)
    pred_out = r'./results/{0}_predict.npz'.format(f_name)

    adata = sc.read_h5ad(adata_f)
    with np.load(pred_out) as data:
        y_pred_label = data['y_pred_gat']
    adata.obs['y_pred_label'] = y_pred_label.astype('str')
    fig, ax = plt.subplots(1, 1, figsize=(centimeter * 10.5, centimeter * 10.5))
    spot_size = 1.5
    if np.max(adata.obsm['spatial']) > 10000:
        spot_size = 80
    if np.max(adata.obsm['spatial']) < 500:
        spot_size = 1.2
    sc.pl.spatial(adata, color="y_pred_label", groups=grp, spot_size=spot_size, frameon=False, show=False, ax=ax)
    titl = 'Niches_ ' + '.'.join(grp)
    ax.set_title(titl)

    plt.savefig(r'./results/{0}_{1}.pdf'.format(f_name, titl), bbox_inches='tight', format='pdf', dpi=300)


def plot_var(f_name, feat_type,var_name):
    adata_f = r'./results/{0}_prep.h5ad'.format(f_name)

    adata = sc.read_h5ad(adata_f)
    adata.obs[var_name]=adata.obsm[feat_type].loc[:,var_name]

    fig, ax = plt.subplots(1, 1, figsize=(centimeter*10.5, centimeter*10.5))
    spot_size = 1.5
    if np.max(adata.obsm['spatial']) > 10000:
        spot_size = 80
    if np.max(adata.obsm['spatial']) < 500:
        spot_size = 1.2
    sc.pl.spatial(adata, color=var_name, spot_size=spot_size, frameon=False, show=False, ax=ax)
    titl = var_name
    ax.set_title(titl)


def plot_cell_frac(adata_f, pred_out, f_name, use_max=True, use_cnt=True):
    adata = sc.read_h5ad(adata_f)
    with np.load(pred_out) as data:
        y_pred_label = data['y_pred_gat']

    cell_frac_df = adata.obsm['cell_frac']
    cell_frac_df['y_pred_label'] = y_pred_label

    if use_max:
        adata.obs['y_pred_label'] = y_pred_label
        if use_cnt:
            niche_ct_avg = adata.obs[['pred_celltype', 'y_pred_label']].groupby(
                ['y_pred_label', 'pred_celltype']).size().unstack()
        else:  # fraction
            niche_ct_avg = adata.obs[['pred_celltype', 'y_pred_label']].groupby(
                ['y_pred_label', 'pred_celltype']).size().unstack()
            niche_ct_avg = niche_ct_avg.div(niche_ct_avg.sum(axis=1), axis=0)
    else:
        niche_ct_avg = cell_frac_df.groupby(['y_pred_label']).mean()

    fig, ax = plt.subplots(figsize=(centimeter * 8.5, centimeter * 7.5))

    niche_ct_avg.plot(kind='bar', stacked=True, ax=ax, edgecolor=None, linewidth=0, width=0.7)
    ax.tick_params(direction='out', length=3, pad=3, width=1)
    leg = ax.legend(loc='center left', bbox_to_anchor=[1, 0.5], ncol=1, frameon=False,
                    markerscale=0.6, labelspacing=0.2, fontsize=5,
                    handleheight=0.8, handlelength=1, title='Annotation')
    leg.get_title().set_fontsize('5')
    sns.despine()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel('Niche')
    if use_cnt:
        ax.set_ylabel('Spot Count')
    else:
        ax.set_ylabel('Spot Fraction')
    ax.set_title(f_name)

    plt.savefig("./results/{0}_niche_cell_fracs.pdf".format(f_name), bbox_inches='tight', format='pdf', dpi=300)


config_rc(dpi=300, font_size=6)
centimeter = 1/2.54


if __name__ == '__main__':

    # print(torch.cuda.is_available())

    # st_adata_names = ['ESCC.E2', 'ICC.LC5.LN.FA4', 'ESCC.E4', 'ICC.LC6.T.FE5', 'CRC.CR1.adata1',
    #                   'ICC.LC4.T.FG2', 'ICC.LC15.B.GK2', 'ICC.LC12.B.GF3', 'LC4-B_FE7_web4_bin50_Cluster2',
    #                   'LC15.B.GK1.info', 'LC04.T.FG2.info']

    # 2023-10-10
    # data see drive
    st_adata_names = ['ESCC.E2', 'ICC.LC5.LN.FA4', 'ESCC.E4', 'ICC.LC6.T.FE5', 'CRC.CR1.adata1',
                      'ICC.LC4.T.FG2', 'ICC.LC15.B.GK2', 'ICC.LC12.B.GF3', 'LC4-B_FE7_web4_bin50_Cluster2',
                      'LC15.adata.moran.filter', 'LC04.adata.moran.filter']

    # # create data
    # for i_data in st_adata_names[:-3]:
    #     print(i_data)
    #     crt_data2(i_data, n_comps=80, min_genes=10, min_cells=1, max_genes_by_cnt=None, regress_out=False)
    # # for 'LC4-B_FE7_web4_bin50_Cluster'
    # crt_data_tmp(n_comps=80, min_genes=10, min_cells=1, max_genes_by_cnt=None, regress_out=False)
    # for i_data in st_adata_names[-2:]:
    #     print(i_data)
    #     crt_data3(i_data, n_comps=80, min_genes=10, min_cells=1, max_genes_by_cnt=None, regress_out=False)

    # 2023-08-22
    # for i_data in st_adata_names[-2:]:
    #     print(i_data)
    #     crt_data4(i_data, min_genes=10, min_cells=1, max_genes_by_cnt=None, regress_out=False)

    # 2023-10-10
    # for i_data in st_adata_names[-2:]:
    #     print(i_data)
    #     crt_data5(i_data)

    idx = 10
    seed_everything(42)
    f_name = st_adata_names[idx]
    # main(f_name)

    method = 'ig'
    res = explain(method, f_name, target=list(range(13)), device='cpu')  #
    np.savez(r'./results/{0}_predlabels_{1}.npz'.format(f_name, method), attributes=res)

    # adata = sc.read_h5ad('./results/' + st_adata_names[idx] + '_prep.h5ad')
    # sc.settings.set_figure_params(vector_friendly=True)
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'aspect': 'equal'})
    # spot_size = 1.5
    # if np.max(adata.obsm['spatial']) > 10000:
    #     spot_size = 80
    # if np.max(adata.obsm['spatial']) < 500:
    #     spot_size = 1.2
    # sc.pl.spatial(adata, color="pred_celltype", spot_size=spot_size, frameon=False, show=False, ax=ax)
    #
    # plt.savefig(r'./results/{0}_cell_type.pdf'.format(st_adata_names[idx]), bbox_inches='tight', format='pdf', dpi=400)






