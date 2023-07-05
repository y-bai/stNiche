#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/16 15:41
@License: (C) Copyright 2013-2023. 
@File: _graph_net.py
@Desc:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv


class GATExt(nn.Module):

    def __init__(self, in_features, dim_hidden, n_hiddens=3, n_z=32, dim_out=20, n_head=2,
                 concat=True, ext_emb=None, embedding='vae', tau=0.001, gate=True):
        """

        GAT model with jointly combination of embedding network

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
        dim_out
            The dimension of the output from the network. `dim_out` = `n_clusters` for clustering.
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

        ext_emb
            The instance of embedding network. If `ext_emb = None`, then no embedding network is used for combination.

        tau
            Weight factor for output of each embedding layer
            when jointly combing with corresponding graph network layer.  Default value = 0.001
        gate
            Indicator if using gating mechanism to jointly combine outputs of each embedding layer
            and corresponding graph network layer. if `gate = True`, then `tau` will not be used. Default value = True.
        embedding
            Specifying embedding network. Could be `vae` or `ae`. Default value = 'vae'.
        """

        super(GATExt, self).__init__()

        self.tau = tau
        self.gate = gate
        self.ext_emb = ext_emb
        self.embedding = embedding

        if isinstance(dim_hidden, int):
            self.dim_hidden = [dim_hidden] * n_hiddens
        else:
            assert isinstance(dim_hidden, list), 'n_hidden should be a type of int or list'
            self.dim_hidden = dim_hidden

        if concat:
            _n_head = n_head
        else:
            # average head
            _n_head = 1

        # first layer
        self.first_gat = GATv2Conv(in_features, out_channels=self.dim_hidden[0], heads=n_head, concat=concat)

        self.gat_net = nn.ModuleList(
            [GATv2Conv(self.dim_hidden[i] * _n_head,
                       out_channels=self.dim_hidden[i + 1],
                       heads=n_head, concat=concat) for i in range(len(self.dim_hidden) - 1)]
        )
        # z-layer
        self.gat_net.append(GATv2Conv(self.dim_hidden[-1] * _n_head,
                                      out_channels=n_z,
                                      heads=1))
        if self.ext_emb is not None:
            assert len(self.ext_emb.emb_net_ens) == len(self.gat_net), \
                'N embedding encoding layers not equal to N graph encoding layers'

        self.final_gat_layer = GATv2Conv(n_z, out_channels=dim_out, heads=1)

        # update gate
        self.gate_layers_u = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.dim_hidden[i] * _n_head * 2, self.dim_hidden[i] * _n_head, bias=False),
                nn.Sigmoid(),
            ) for i in range(len(self.dim_hidden))]
        )

        # reset gate
        self.gate_layers_r = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.dim_hidden[i] * _n_head * 2, self.dim_hidden[i] * _n_head, bias=False),
                nn.Sigmoid(),
            ) for i in range(len(self.dim_hidden))]
        )

        # self.gate_layers2 = nn.ModuleList(
        #     [nn.Sequential(
        #         nn.Linear(self.dim_hidden[i] * _n_head * 2, self.dim_hidden[i] * _n_head, bias=False),
        #         # nn.LeakyReLU(negative_slope=0.2),
        #         nn.Tanh()
        #     ) for i in range(len(self.dim_hidden))]
        # )

        self.gate_layers_comb = nn.ModuleList(
            [
                nn.Linear(self.dim_hidden[i] * _n_head * 2, self.dim_hidden[i] * _n_head, bias=False)
                for i in range(len(self.dim_hidden))
            ]
        )

        self.gate_layers_u.append(nn.Sequential(
            nn.Linear(n_z * 2, n_z, bias=False),
            nn.Sigmoid(),
        ))

        self.gate_layers_r.append(nn.Sequential(
            nn.Linear(n_z * 2, n_z, bias=False),
            nn.Sigmoid(),
        ))

        self.gate_layers_comb.append(nn.Linear(n_z * 2, n_z, bias=False))

    def forward(self, x, adj):

        # embedding net
        if self.ext_emb is not None:
            if self.embedding == 'vae':
                x_rec, z, mu, log_var, emd_enc_layers = self.ext_emb(x)
            if self.embedding == 'ae':
                x_rec, z, emd_enc_layers = self.ext_emb(x)

        # graph net
        h = self.first_gat(x, adj)
        # h = F.leaky_relu(h, negative_slope=0.2)   # DEC not using such activate function

        for i, _gat in enumerate(self.gat_net):
            if self.ext_emb is not None:
                if self.gate:
                    temp = torch.cat((h, emd_enc_layers[i]), 1)  # first layer torch.Size([61081, 512])
                    update_gate = self.gate_layers_u[i](temp)
                    reset_gate = self.gate_layers_r[i](temp)
                    h_tilde = F.tanh(self.gate_layers_comb[i](
                            torch.cat((reset_gate * h, emd_enc_layers[i]), 1)
                    ))
                    h = h_tilde * update_gate + h * (1 - update_gate)
                else:
                    h = h * (1 - self.tau) + emd_enc_layers[i] * self.tau
            h = _gat(h, adj)
            # h = F.leaky_relu(h, negative_slope=0.2)
        # z_layer
        if self.ext_emb is not None:
            if self.gate:
                temp = torch.cat((h, z), 1)  #
                update_gate = self.gate_layers_u[-1](temp)
                reset_gate = self.gate_layers_r[-1](temp)
                h_tilde = F.tanh(self.gate_layers_comb[-1](
                    torch.cat((reset_gate * h, z), 1)
                ))
                h = h_tilde * update_gate + h * (1 - update_gate)
            else:

                h = h * (1 - self.tau) + z * self.tau

        # gat_z = h

        if self.ext_emb is None:
            return h   # hidden representation
        else:
            h = self.final_gat_layer(h, adj)  # final predict
            h = F.softmax(h, dim=1)

            if self.embedding == 'vae':
                return h, x_rec, z, mu, log_var
            if self.embedding == 'ae':
                return h, x_rec, z


