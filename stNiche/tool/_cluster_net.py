#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/16 13:21
@License: (C) Copyright 2013-2023. 
@File: _cluster_net.py
@Desc:

"""


import torch
import torch.nn as nn

from ..embed import VAE, AE
from ..embed import GATExt
from ..loss import cal_dec_q


class GraphEmbNetCluster(nn.Module):
    """

    Unsupervised deep clustering using graph network and embedding network

        Graph network: GAT
        Embedding network: VAE (vanilla VAE or beta VAE), or AE (vanilla AE or contractive AE)

    """
    def __init__(self, in_features, dim_hidden, n_clusters, n_hiddens=3, n_z=32, n_head=2, concat=True,
                 tau=0.001, gate=True, embedding='vae', pretrain_path='emb_net.ckpt'):

        """

        Parameters
        ----------
        in_features
            The number of input features of raw data X
        dim_hidden
            The number of neurons for each hidden layer. It can be a list of int or int.
                For example: `dim_hidden = [256, 512, 256, 64]` means there are 4 hidden layers, with the number of
                neurons specified.
                if `dim_hidden = 64` means all layers having the same number of neurons.
        n_clusters
            The number of clusters could be inference.
        n_hiddens
            The number of hidden layers. When `dim_hidden` is int value, then `n_hiddens` specifies the number of
            hidden layers.  Default value = 3.
        n_z
            The dimension of z representation/output from z layer. Default value = 32.
        n_head
            The number of heads for attention mechanism (ie. GAT). Default value = 2.
        concat
            Indicator if the outputs of multi-heads would be concatenated or be calculating average value.
            `concat = False` means calculating average value. Default value = True.
            See Also
            https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html
        tau
            Weight factor for output of each embedding layer
            when jointly combing with corresponding graph network layer.  Default value = 0.001
        gate
            Indicator if using gating mechanism to jointly combine outputs of each embedding layer
            and corresponding graph network layer. if `gate = True`, then `tau` will not be used. Default value = True.
        embedding
            Specifying embedding network. Could be `vae` or `ae`. Default value = 'vae'.
        pretrain_path
            Pre-trained embedding network checkpoint path.
        """

        super(GraphEmbNetCluster, self).__init__()

        if isinstance(dim_hidden, int):
            self.dim_hidden = [dim_hidden] * n_hiddens
        else:
            assert isinstance(dim_hidden, list), 'n_hidden should be a type of int or list'
            self.dim_hidden = dim_hidden

        self.embedding = embedding
        self.emb_net = None
        if self.embedding == 'vae':
            self.emb_net = VAE(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z, n_head=n_head,
                               concat=concat)
            if pretrain_path is None:
                print('warning: pretrain_path not specify')
            else:
                self.emb_net.load_state_dict(torch.load(pretrain_path))

        if self.embedding == 'ae':
            self.emb_net = AE(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z, n_head=n_head,
                              concat=concat)
            if pretrain_path is None:
                print('warning: pretrain_path not specify')
            else:
                self.emb_net.load_state_dict(torch.load(pretrain_path))

        assert self.emb_net, "embedding network is not specified, try embedding='vae'"
        self.graph_net = GATExt(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z,
                                dim_out=n_clusters, n_head=n_head, concat=concat,
                                ext_emb=self.emb_net, embedding=embedding, tau=tau, gate=gate)

        self.v = 1  # degree to calculate q
        self.n_z = n_z

        # cluster layer
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, data):
        x = data.x
        adj = data.edge_index
        if self.embedding == 'vae':
            gat_h, emb_recons, emb_z, mu, log_var = self.graph_net(x, adj)
            q = cal_dec_q(emb_z, self.cluster_layer, self.v)
            # q = cal_dec_q(gat_h, self.cluster_layer, self.v)
            return gat_h, emb_recons, emb_z, q, mu, log_var

        if self.embedding == 'ae':
            gat_h, emb_recons, emb_z = self.graph_net(x, adj)
            q = cal_dec_q(emb_z, self.cluster_layer, self.v)
            # q = cal_dec_q(gat_h, self.cluster_layer, self.v)
            return gat_h, emb_recons, emb_z, q


class GraphNetCluster(nn.Module):
    """

        Unsupervised deep clustering only using graph network

            Graph network: GAT

        """
    def __init__(self, in_features, dim_hidden, n_clusters, n_hiddens=3, n_z=32, n_head=2, concat=True):

        """

        Parameters
        ----------
        in_features
            The number of input features of raw data X
        dim_hidden
            The number of neurons for each hidden layer. It can be a list of int or int.
                For example: `dim_hidden = [256, 512, 256, 64]` means there are 4 hidden layers, with the number of
                neurons specified.
                if `dim_hidden = 64` means all layers having the same number of neurons.
        n_clusters
            The number of clusters could be inference.
        n_hiddens
            The number of hidden layers. When `dim_hidden` is int value, then `n_hiddens` specifies the number of
            hidden layers.  Default value = 3.
        n_z
            The dimension of z representation/output from z layer. Default value = 32.
        n_head
            The number of heads for attention mechanism (ie. GAT). Default value = 2.
        concat
            Indicator if the outputs of multi-heads would be concatenated or be calculating average value.
            `concat = False` means calculating average value. Default value = True.
            See Also
            https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html
        """

        super(GraphNetCluster, self).__init__()
        if isinstance(dim_hidden, int):
            self.dim_hidden = [dim_hidden] * n_hiddens
        else:
            assert isinstance(dim_hidden, list), 'n_hidden should be a type of int, list or tuple'
            self.dim_hidden = dim_hidden

        # Graph Net
        self.graph_net = GATExt(in_features, dim_hidden, n_z=n_z, dim_out=n_clusters, n_hiddens=n_hiddens,
                                n_head=n_head, concat=concat, ext_emb=None)

        # cluster layer
        self.v = 1
        self.n_z = n_z

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, data):
        x = data.x
        adj = data.edge_index

        gat_h = self.graph_net(x, adj)
        # print(gat_h.size())
        # print(self.cluster_layer.size())
        q = cal_dec_q(gat_h, self.cluster_layer, self.v)

        return gat_h, q,


class EmbNetCluster(nn.Module):
    """

    Unsupervised deep clustering only using embedding network

        Embedding network: VAE (vanilla VAE or beta VAE), or AE (vanilla AE or contractive AE)

    """
    def __init__(self, in_features, dim_hidden, n_clusters, n_hiddens=3, n_z=32, n_head=2, concat=True,
                 embedding='vae', pretrain_path='emb_net.ckpt'):

        """

        Parameters
        ----------
        in_features
            The number of input features of raw data X
        dim_hidden
            The number of neurons for each hidden layer. It can be a list of int or int.
                For example: `dim_hidden = [256, 512, 256, 64]` means there are 4 hidden layers, with the number of
                neurons specified.
                if `dim_hidden = 64` means all layers having the same number of neurons.
        n_clusters
            The number of clusters could be inference.
        n_hiddens
            The number of hidden layers. When `dim_hidden` is int value, then `n_hiddens` specifies the number of
            hidden layers.  Default value = 3.
        n_z
            The dimension of z representation/output from z layer. Default value = 32.
        n_head
            The number of heads for attention mechanism (ie. GAT). Default value = 2.

            Here, `n_head` is used for making sure the dimension is tha same as
            graph network when using multi-head attention.

        concat
            Indicator if the outputs of multi-heads would be concatenated or be calculating average value.
            `concat = False` means calculating average value. Default value = True.
            See Also
            https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html

            Here, `concat` is used for making sure the dimension is the same as
            graph network when using multi-head attention.

        embedding
            Specifying embedding network. Could be `vae` or `ae`. Default value = 'vae'.
        pretrain_path
            Pre-trained embedding network checkpoint path.

        """

        super(EmbNetCluster, self).__init__()

        if isinstance(dim_hidden, int):
            self.dim_hidden = [dim_hidden] * n_hiddens
        else:
            assert isinstance(dim_hidden, (list, tuple)), 'n_hidden should be a type of int, list or tuple'
            self.dim_hidden = dim_hidden

        self.embedding = embedding
        self.emb_net = None
        if self.embedding == 'vae':
            self.emb_net = VAE(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z, n_head=n_head,
                               concat=concat)
            self.emb_net.load_state_dict(torch.load(pretrain_path))

        if self.embedding == 'ae':
            self.emb_net = AE(in_features, dim_hidden, n_hiddens=n_hiddens, n_z=n_z, n_head=n_head,
                              concat=concat)
            self.emb_net.load_state_dict(torch.load(pretrain_path))

        assert self.emb_net is not None, "embedding network is not specified, try embedding='vae'"

        self.v = 1  # degree to calculate q
        self.n_z = n_z

        # cluster layer
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):

        if self.embedding == 'vae':
            x_rec, z, mu, log_var, _ = self.emb_net(x)
            q = cal_dec_q(z, self.cluster_layer, self.v)
            return x_rec, z, q, mu, log_var

        if self.embedding == 'ae':
            x_rec, z, _ = self.emb_net(x)
            q = cal_dec_q(z, self.cluster_layer, self.v)
            # print(z.size())   # torch.Size([2048, 32])
            # print(q.size())   # torch.Size([2048, 10])
            # print(self.cluster_layer.size())   # torch.Size([10, 32])

            return x_rec, z, q
