#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/16 15:41
@License: (C) Copyright 2013-2023. 
@File: _embed_net.py
@Desc:

"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_features, dim_hidden, n_hiddens=3, n_z=32, n_head=2, concat=True):
        """
        VAE embedding embed

        Parameters
        ----------
        in_features
            The number of input features of raw data X
        dim_hidden
            The number of neurons for each hidden layer. It can be a list of int or int.
                For example: `dim_hidden = [256, 512, 256, 64]` means there are 4 hidden layers, with the number of
                neurons specified.
                if `dim_hidden = 64` means all layers having the same number of neurons.
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
        """

        super(VAE, self).__init__()

        self.n_z = n_z

        if isinstance(dim_hidden, int):
            self.dim_hidden = [dim_hidden] * n_hiddens
        else:
            assert isinstance(dim_hidden, list), 'n_hidden should be a type of int or list '
            self.dim_hidden = dim_hidden

        if concat:
            _n_head = n_head
        else:
            # average head
            _n_head = 1

        # we use VAE to embed (encoder)
        # first layer
        self.emb_net_ens = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features, self.dim_hidden[0] * _n_head),
            nn.BatchNorm1d(self.dim_hidden[0] * _n_head),
            nn.LeakyReLU(0.2)
        )])

        for i in range(len(self.dim_hidden) - 1):
            self.emb_net_ens.append(nn.Sequential(
                nn.Linear(self.dim_hidden[i] * _n_head,
                          self.dim_hidden[i + 1] * _n_head),
                nn.BatchNorm1d(self.dim_hidden[i + 1] * _n_head),
                # nn.Dropout(self.dropout),
                nn.LeakyReLU(0.2)
            ))

        # z layer
        self.z_mu = nn.Linear(self.dim_hidden[-1] * _n_head, self.n_z)
        self.z_log_var = nn.Linear(self.dim_hidden[-1] * _n_head, self.n_z)

        # vae decoder
        emb_net_decoder_list = []  # we use VAE to embed (decoder)
        invs_hidden = [self.n_z] + self.dim_hidden[::-1]
        for i in range(len(invs_hidden) - 1):
            emb_net_decoder_list.extend([
                nn.Linear(invs_hidden[i] if i == 0 else invs_hidden[i] * _n_head,
                          invs_hidden[i + 1] * _n_head),
                nn.BatchNorm1d(invs_hidden[i + 1] * _n_head),
                # nn.Dropout(self.dropout),
                nn.LeakyReLU(0.2)
            ])
        self.emb_net_decoder = nn.Sequential(*emb_net_decoder_list)

        # reconstruct
        self.x_bar = nn.Linear(invs_hidden[-1] * _n_head, in_features)
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
            # https://discuss.pytorch.org/t/kld-loss-goes-nan-during-vae-training/42305/
            # n = module.in_features
            # y = 1.0/np.sqrt(n)
            # # module.weight.data.uniform_(-0.08, 0.08)
            # module.weight.data.uniform_(-y, y)
            # module.bias.data.fill_(0)

        if isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight.data, 1)
            nn.init.constant_(module.bias.data, 0)

    def _sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):

        emd_enc_layers = []
        for _enc in self.emb_net_ens:
            x = _enc(x)
            emd_enc_layers.append(x)

        # sampling z
        mu = self.z_mu(x)
        log_var = self.z_log_var(x)
        z = self._sampling(mu, log_var)

        # decode
        h_in = self.emb_net_decoder(z)
        # reconstruct
        x_rec = self.x_bar(h_in)

        return x_rec, z, mu, log_var, emd_enc_layers


class AE(nn.Module):
    def __init__(self, in_features, dim_hidden, n_hiddens=3, n_z=32, n_head=2, concat=True):
        """

        AE embedding embed

        Parameters
        ----------
        in_features
            The number of input features of raw data X
        dim_hidden
            The number of neurons for each hidden layer. It can be a list of int or int.
                For example: `dim_hidden = [256, 512, 256, 64]` means there are 4 hidden layers, with the number of
                neurons specified.
                if `dim_hidden = 64` means all layers having the same number of neurons.
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
        """

        super(AE, self).__init__()

        self.n_z = n_z

        if isinstance(dim_hidden, int):
            self.dim_hidden = [dim_hidden] * n_hiddens
        else:
            assert isinstance(dim_hidden, list), 'n_hidden should be a type of int or list '
            self.dim_hidden = dim_hidden

        if concat:
            _n_head = n_head
        else:
            # average head
            _n_head = 1

        # we use VAE to embed (encoder)
        # first layer
        self.emb_net_ens = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features, self.dim_hidden[0] * _n_head),
            nn.BatchNorm1d(self.dim_hidden[0] * _n_head),
            nn.LeakyReLU(0.2)
        )])

        for i in range(len(self.dim_hidden) - 1):
            self.emb_net_ens.append(nn.Sequential(
                nn.Linear(self.dim_hidden[i] * _n_head,
                          self.dim_hidden[i + 1] * _n_head),
                nn.BatchNorm1d(self.dim_hidden[i + 1] * _n_head),
                # nn.Dropout(self.dropout),
                nn.LeakyReLU(0.2)
            ))

        # z layer
        self.z_layer = nn.Sequential(
            nn.Linear(self.dim_hidden[-1] * _n_head, self.n_z),
            nn.BatchNorm1d(self.n_z),
            nn.LeakyReLU(0.2)
        )

        # vae decoder
        emb_net_decoder_list = []  # we use VAE to embed (decoder)
        invs_hidden = [self.n_z] + self.dim_hidden[::-1]
        for i in range(len(invs_hidden) - 1):
            emb_net_decoder_list.extend([
                nn.Linear(invs_hidden[i] if i == 0 else invs_hidden[i] * _n_head,
                          invs_hidden[i + 1] * _n_head),
                nn.BatchNorm1d(invs_hidden[i + 1] * _n_head),
                nn.LeakyReLU(0.2)
            ])
        self.emb_net_decoder = nn.Sequential(*emb_net_decoder_list)
        # reconstruct
        self.x_bar = nn.Linear(invs_hidden[-1] * _n_head, in_features)
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
            # https://discuss.pytorch.org/t/kld-loss-goes-nan-during-vae-training/42305/
            # n = module.in_features
            # y = 1.0/np.sqrt(n)
            # # module.weight.data.uniform_(-0.08, 0.08)
            # module.weight.data.uniform_(-y, y)
            # module.bias.data.fill_(0)

        if isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight.data, 1)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):

        emd_enc_layers = []
        for _enc in self.emb_net_ens:
            x = _enc(x)
            emd_enc_layers.append(x)

        z = self.z_layer(x)

        # decode
        h_in = self.emb_net_decoder(z)
        # reconstruct
        x_rec = self.x_bar(h_in)

        return x_rec, z, emd_enc_layers
