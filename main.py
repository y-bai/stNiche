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
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

import stNiche as stn


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


def crt_data():
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

    # adata filter and preprocess
    adata = stn.pp.adata_prep(adata,
                              min_genes=100, min_cells=3, max_genes_by_cnt=2500, max_pct_mt=5, norm=True,
                              target_sum=1e4, log1=True, n_hvg=2500, regress_out=True, scale=False)
    print('After preprocess:\n', adata)

    # adata dimension reduction
    n_comps = 20
    sc.tl.pca(adata, n_comps=n_comps, random_state=42)

    # save preprocessed data
    adata.write_h5ad(r'./results/prep_adata.pt')

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
    print(spatial.shape)
    adj, _ = stn.utils.crt_graph(spatial, mode='knn_graph', top_k=8)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    print(edge_index)
    # torch.geometric dataset
    g_data = Data(x=torch.tensor(x, dtype=torch.float32),
                  edge_index=edge_index)

    torch.save(g_data, r'./results/graph_data_knn24.pt')


def plot_out(adata, y_pred, z, out_fname):
    sc.settings.set_figure_params(vector_friendly=True)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), subplot_kw={'aspect': 'equal'})
    axs = axs.ravel()

    spot_size = 1.5
    adata.obs['y_pred'] = y_pred.astype('str')
    sc.pl.spatial(adata, color="y_pred", spot_size=spot_size, frameon=False,
                  show=False, ax=axs[0])

    adata.obsm['niche_emb'] = z
    sc.pp.neighbors(adata, use_rep='niche_emb', key_added='niche_emb_neighbor')
    sc.tl.umap(adata, neighbors_key='niche_emb_neighbor')
    sc.pl.umap(adata, color=["y_pred"], wspace=0.4, neighbors_key='niche_emb_neighbor', show=False, ax=axs[1])

    plt.savefig(out_fname, bbox_inches='tight', format='pdf', dpi=400)


def main():

    # load data
    g_data = torch.load(r'./results/graph_data_knn24.pt')
    adata = sc.read_h5ad(r'./results/prep_adata.pt')

    x = g_data.x.numpy()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    concat = True
    in_features = x.shape[-1]
    print(in_features)
    n_hiddens = 1
    n_head = 2
    n_clusters = 12  # increase this with lr
    dim_hidden = [256, 512, 256, 64]  # [512, 64]
    n_z = 32

    #
    learning_rate = 0.01  # 0.01

    max_epochs = 50
    l2_weight = 0.000001
    batch_size = 1024 * 2

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

    embedding_model_path = r'./results/emb_net_{0}.ckpt'.format(model_type)

    #################################################################################################
    # embedding pretrain
    if embedding == 'vae':
        emb_model = stn.model.VAE(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z, n_head=n_head, concat=concat)
    if embedding == 'ae':
        emb_model = stn.model.AE(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z, n_head=n_head, concat=concat)
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
    # graph_cluster = stn.model.GraphNetCluster(in_features, dim_hidden, n_clusters,
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
    # emb_cluster = stn.model.EmbNetCluster(in_features, dim_hidden, n_clusters, n_hiddens=n_hiddens,
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

    graph_emb_model = stn.model.GraphEmbNetCluster(
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
    np.savez(r'./results/predict.npz', y_pred_label_q=y_pred_q,
             y_pred_gat=y_pred_gat, gat_h=gat_h, emb_z=emb_z)
    # f = np.load(r'./results/predict.npz')
    plot_out(adata, y_pred_gat, emb_z, r'./results/{0}_final_pred_{1}.pdf'.format(model_type, combine))
    # plot_out(adata, y_pred_q, emb_z, r'./results/{0}_final_pred2_{1}.pdf'.format(model_type, combine))
    # plot_out(adata, y_pred_gat, gat_h, r'./results/{0}_final_pred3_{1}.pdf'.format(model_type, combine))


if __name__ == '__main__':
    seed_everything(42)
    # crt_data()
    main()

    # adata = sc.read_h5ad(r'./results/prep_adata.pt')
    # sc.settings.set_figure_params(vector_friendly=True)
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'aspect': 'equal'})
    # spot_size = 1.5
    # sc.pl.spatial(adata, color="pred_celltype", spot_size=spot_size, frameon=False, show=False, ax=ax)
    #
    # plt.savefig(r'./results/cell_type.pdf', bbox_inches='tight', format='pdf', dpi=400)






