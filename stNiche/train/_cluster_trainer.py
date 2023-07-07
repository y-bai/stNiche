#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/6/21 11:28
@License: (C) Copyright 2013-2023. 
@File: _cluster_trainer.py
@Desc:

"""

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, StepLR
from ..utils import EmbDataset
from ..loss import vae_loss, ae_loss, dec_loss, target_dist


class GraphEmbNetClusterTrainer:
    """

    Trainer for joint combination of graph network and embedding network clustering

    """
    def __init__(self, model, init_cluster_center, init_cluster_label,
                 tol=0.001, embedding='vae', model_type='vanilla_vae',
                 alpha_emb_recons=0.01, alpha_emb_cluster=0.001, alpha_gat_cluster=5.0,
                 beta=0.001, gamma=0.0001, c_max=25, lam=1e-4,
                 device='cuda', learning_rate=0.01, max_epochs=50, l2_weight=0.00001):
        """

        Parameters
        ----------
        model
            Instance of `GraphEmbNetCluster`, see `embed._cluster_net.GraphEmbNetCluster`
        init_cluster_center
            2d np.array of initial clustering center, with shape (n_clusters, n_z)
        init_cluster_label
            np.array of initial clustering labels, with shape (n_sample,)
        tol
            tolerance of difference current updated predicted labels and previous predicted labels. Deprecated.

        embedding
             Specifying embedding network. Could be `vae` or `ae`. Default value = 'vae'.

        model_type
            type of embedding networks.
                If `embedding = 'vae'`, then `model_type` could be `vanilla_vae` or `beta_vae`.
                If `embedding = 'ae'`, then `model_type` could be `vanilla_ae` or `contractive_ae`.

        alpha_emb_recons
            weight factor for embedding network loss (eg. reconstruction loss). Default value = 0.01.
        alpha_emb_cluster
            weight factor for deep embedding clustering loss from embedding network. Default value = 0.001.
        alpha_gat_cluster
            weight factor for deep embedding clustering loss from graph network. Default value = 5.0

        beta
            weight used for vanilla VAE KL loss, default value = 0.001,
            see see `loss._loss.vae_loss`
        gamma
            weight used for beta VAE KL loss, default value = 0.0001,
            see see `loss._loss.vae_loss`
        c_max
            maximum value of controllable value for beta VAE, default value = 25.0,
            see see `loss._loss.vae_loss`

        lam
            weight factor for contractive loss, default value = 1e-4

        device
            device used for the embed
        learning_rate
            learning rate, default value = 0.01
        max_epochs
            the maximum number of epochs, default value = 50
        l2_weight
            weight factor for L2 regularization of embed weights, default value = 0.00001
        """

        self.alpha1 = alpha_emb_recons
        self.alpha2 = alpha_emb_cluster
        self.alpha3 = alpha_gat_cluster
        self.beta = beta
        self.gamma = gamma
        self.c_max = c_max
        self.lam = lam

        self.y_pred_last = init_cluster_label
        self.tol = tol
        self.max_epochs = max_epochs

        self.embedding = embedding
        self.model_type = model_type

        self.device = device
        self.model = model.to(self.device)
        self.model.cluster_layer.data = torch.tensor(init_cluster_center).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_weight)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)  # 10
        self.scheduler = StepLR(self.optimizer, step_size=max_epochs // 4, gamma=0.01)

    def fit(self, data):
        """

        train embed

        Parameters
        ----------
        data
            PyG data format, with node feature matrix and edge_index.
        """
        data = data.to(self.device)
        for epoch in range(self.max_epochs):

            if epoch % 1 == 0:
                with torch.no_grad():
                    if self.embedding == 'vae':
                        _, _, _, tmp_q, _, _ = self.model(data)
                    if self.embedding == 'ae':
                        _, _, _, tmp_q = self.model(data)

                    p = target_dist(tmp_q.detach())
                    p = p.to(self.device)

                    # y_pred = tmp_q.detach().cpu().numpy().argmax(1)
                    #
                    # delta_label = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
                    # self.y_pred_last = y_pred
                    #
                    # if epoch > 0 and delta_label < self.tol:
                    #     print('delta_label ', delta_label, '< tol ', self.tol)
                    #     print('Reached tolerance threshold. Stopping training.')
                    #     break

            self.model.train()
            self.optimizer.zero_grad()

            if self.embedding == 'vae':
                gat_h, emb_recons, emb_z, q, mu, log_var = self.model(data)

                emb_loss = vae_loss(data.x, emb_recons, mu, log_var, self.max_epochs, epoch, self.device,
                                    beta=self.beta, gamma=self.gamma, c_max=self.c_max,
                                    model_type=self.model_type)

            if self.embedding == 'ae':
                gat_h, emb_recons, emb_z, q = self.model(data)
                w = self.model.emb_net.z_layer[0].weight
                emb_loss = ae_loss(data.x, emb_recons, model_type=self.model_type, w=w, h=emb_z, lam=self.lam)

            # print(q.size())   # torch.Size([61081, 10])
            # print(p.size())   # torch.Size([61081, 10])
            # print(gat_h.size())

            cluster_emb = dec_loss(q, p)
            cluster_gat = dec_loss(gat_h, p)

            loss = self.alpha1 * emb_loss['total_loss'] + self.alpha2 * cluster_emb + self.alpha3 * cluster_gat
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=self.embed.parameters(), max_norm=0.1, norm_type=2.0)
            self.optimizer.step()
            print(
                'Epoch [{}/{}], total Loss: {:.4f}, '
                'recons Loss: {:.4f}, KL emb_qp: {:.4f}, KL gat_qp: {:.4f}'.format(
                    epoch + 1, self.max_epochs, loss.item(), emb_loss['recons_loss'].item(),
                    cluster_emb.item(), cluster_gat.item()))

            self.scheduler.step()

    def predict(self, data):
        """

        predict cluster label

        Parameters
        ----------
        data
            PyG data format, with node feature matrix and edge_index.
        Returns
        -------
        y_pred_q
            cluster labels predicted by embedding network, np.array() format

        y_pred_gat
            cluster labels predicted by graph network, np.array() format. Recommendation.

        gat_h.detach().cpu().numpy()
            sample representations by output from graph network.

        emb_z.detach().cpu().numpy()
                 sample representations by z from embedding network.

        """
        self.model.eval()
        with torch.no_grad():
            if self.embedding == 'vae':
                gat_h, emb_recons, emb_z, q, mu, log_var = self.model(data)
            if self.embedding == 'ae':
                gat_h, emb_recons, emb_z, q = self.model(data)

            y_pred_q = q.detach().cpu().numpy().argmax(1)
            y_pred_gat = gat_h.detach().cpu().numpy().argmax(1)
        return y_pred_q, y_pred_gat, gat_h.detach().cpu().numpy(), emb_z.detach().cpu().numpy()

    def save_model(self, path_fname):
        """

        Parameters
        ----------
        path_fname
            file used for saving the embed

        """
        print('saving embedding net...')
        torch.save(self.model.state_dict(), path_fname)


class GraphNetClusterTrainer:
    """

    Trainer for only graph network clustering

    """
    def __init__(self, model, init_cluster_center, init_cluster_label, tol=0.001,
                 device='cuda', learning_rate=0.01, max_epochs=50, l2_weight=0.00001):

        """

        Parameters
        ----------
        model
            Instance of `GraphNetCluster`, see `embed._cluster_net.GraphNetCluster`
        init_cluster_center
            2d np.array of initial clustering center, with shape (n_clusters, n_z)
        init_cluster_label
            np.array of initial clustering labels, with shape (n_sample,)

        tol
            tolerance of difference current updated predicted labels and previous predicted labels. Deprecated.

        device
            device used for the embed
        learning_rate
            learning rate, default value = 0.01
        max_epochs
            the maximum number of epochs, default value = 50
        l2_weight
            weight factor for L2 regularization of embed weights, default value = 0.00001
        """

        self.device = device
        self.max_epochs = max_epochs
        self.tol = tol

        self.y_pred_last = init_cluster_label

        self.model = model.to(self.device)  # graph net cluster
        self.model.cluster_layer.data = torch.tensor(init_cluster_center).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_weight)

    def fit(self, data):
        """

        train embed

        Parameters
        ----------
        data
            PyG data format, with node feature matrix and edge_index.

        """
        data = data.to(self.device)
        for epoch in range(self.max_epochs):
            if epoch % 1 == 0:
                with torch.no_grad():
                    _, tmp_q = self.model(data)
                    p = target_dist(tmp_q.detach())
                    # y_pred = tmp_q.detach().cpu().numpy().argmax(1)
                    # delta_label = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
                    # self.y_pred_last = y_pred
                    #
                    # if epoch > 0 and delta_label < self.tol:
                    #     print('delta_label ', delta_label, '< tol ', self.tol)
                    #     print('Reached tolerance threshold. Stopping training.')
                    #     break

            self.model.train()
            self.optimizer.zero_grad()
            gat_pred, q = self.model(data)

            loss = dec_loss(q, p.to(self.device))

            loss.backward()
            self.optimizer.step()

            print('Epoch [{}/{}], total Loss: {:.4f}'.format(epoch + 1, self.max_epochs, loss.item()))

    def predict(self, data):
        """

        predict cluster label

        Parameters
        ----------
        data
            PyG data format, with node feature matrix and edge_index.

        Returns
        -------
        y_pred_q
            cluster labels predicted by graph network, np.array() format.

        gat_pred.detach().cpu().numpy()
             sample representations by output from graph network.

        """
        self.model.eval()
        with torch.no_grad():
            gat_pred, q = self.model(data)
            y_pred_q = q.detach().cpu().numpy().argmax(1)
        return y_pred_q, gat_pred.detach().cpu().numpy()

    def save_model(self, path_fname):
        """

        Parameters
        ----------
        path_fname
            file used for saving the embed

        """
        print('saving embedding net...')
        torch.save(self.model.state_dict(), path_fname)


class EmbNetClusterTrainer:
    """

    Trainer for only embedding network clustering

    ref: https://github.com/XifengGuo/IDEC/blob/master/IDEC.py

    """
    def __init__(self, model, init_cluster_center, init_cluster_label,
                 tol=0.001, batch_size=1024 * 2, embedding='vae', model_type='vanilla_vae',
                 alpha=1, beta=0.001, gamma=0.0001, c_max=25.0, lam=1e-4,
                 device='cuda', learning_rate=0.01, max_epochs=50, l2_weight=0.00001):
        """


        Parameters
        ----------
        model
            embed
            Instance of `EmbNetCluster`, see `embed._cluster_net.EmbNetCluster`

        init_cluster_center
            2d np.array of initial clustering center, with shape (n_clusters, n_z)
        init_cluster_label
            np.array of initial clustering labels, with shape (n_sample,)

        tol
            tolerance of difference current updated predicted labels and previous predicted labels. Deprecated.
        batch_size
            batch size during embed training, default value = 1024 * 2

        embedding
             Specifying embedding network. Could be `vae` or `ae`. Default value = 'vae'.

        model_type
            type of embedding networks.
                If `embedding = 'vae'`, then `model_type` could be `vanilla_vae` or `beta_vae`.
                If `embedding = 'ae'`, then `model_type` could be `vanilla_ae` or `contractive_ae`.

        alpha
            weight factor for deep embedding clustering loss from embedding network. Default value = 1.

        beta
            weight used for vanilla VAE KL loss, default value = 0.001,
            see see `loss._loss.vae_loss`
        gamma
            weight used for beta VAE KL loss, default value = 0.0001,
            see see `loss._loss.vae_loss`
        c_max
            maximum value of controllable value for beta VAE, default value = 25.0,
            see see `loss._loss.vae_loss`

        lam
            weight factor for contractive loss, default value = 1e-4

        device
            device used for the embed
        learning_rate
            learning rate, default value = 0.01
        max_epochs
            the maximum number of epochs, default value = 50
        l2_weight
            weight factor for L2 regularization of embed weights, default value = 0.00001

        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c_max = c_max

        self.lam = lam

        self.device = device
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.y_pred_last = init_cluster_label
        self.tol = tol

        self.embedding = embedding
        self.model_type = model_type

        self.model = model.to(self.device)  # _cluster_net
        self.model.cluster_layer.data = torch.tensor(init_cluster_center).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_weight)

    def fit(self, x):
        """

        training embedding network

        Parameters
        ----------
        x
            raw input 2d np.array with shape (N_sample, N_feature)

        """

        dataloader = torch.utils.data.DataLoader(EmbDataset(x), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epochs):

            if epoch % 3 == 0:
                with torch.no_grad():
                    if self.embedding == 'vae':
                        _, _, tmp_q, _, _ = self.model(torch.tensor(x).to(self.device))
                    if self.embedding == 'ae':
                        _, _, tmp_q = self.model(torch.tensor(x).to(self.device))
                    p = target_dist(tmp_q.detach())

                    # y_pred = tmp_q.detach().cpu().numpy().argmax(1)
                    # delta_label = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
                    # self.y_pred_last = y_pred
                    #
                    # if epoch > 0 and delta_label < self.tol:
                    #     print('delta_label ', delta_label, '< tol ', self.tol)
                    #     print('Reached tolerance threshold. Stopping training.')
                    #     break

            total_step = len(dataloader)
            self.model.train()
            for i_step, (batch, idxs) in enumerate(dataloader):
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                if self.embedding == 'vae':
                    x_rec, z, q, mu, log_var = self.model(batch)  #

                    train_loss = vae_loss(batch, x_rec, mu, log_var, self.max_epochs, epoch, self.device,
                                              beta=self.beta, gamma=self.gamma, c_max=self.c_max,
                                              model_type=self.model_type)

                if self.embedding == 'ae':
                    x_rec, z, q = self.model(batch)
                    w = self.model.emb_net.z_layer[0].weight
                    train_loss = ae_loss(batch, x_rec, model_type=self.model_type, w=w, h=z, lam=self.lam)

                cluster_loss = dec_loss(q, p[idxs].to(self.device))

                loss = train_loss['total_loss'] + self.alpha * cluster_loss

                loss.backward()
                self.optimizer.step()

                if (i_step + 1) % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], total Loss: {:.4f}, emb Loss: {:.4f}, '
                          'KL_pq: {:.4f}'
                          .format(epoch + 1, self.max_epochs, i_step + 1, total_step,
                                  loss.item(), train_loss['total_loss'].item(), cluster_loss.item()))

    def predict(self, x):
        """

        predict cluster label

        Parameters
        ----------
        x
            raw input 2d np.array with shape (N_sample, N_feature)

        Returns
        -------
        z.detach().cpu().numpy()
            sample representation learned by the embedding network
        y_pred
            cluster labels predicted by the embedding network, np.array() format.

        """
        x = torch.tensor(x)
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            if self.embedding == 'vae':
                x_rec, z, q, mu, log_var = self.model(x)
            if self.embedding == 'ae':
                x_rec, z, q = self.model(x)

            y_pred = q.detach().cpu().numpy().argmax(1)
            return z.detach().cpu().numpy(), y_pred

    def save_model(self, path_fname):
        """

        Parameters
        ----------
        path_fname
            file used for saving the embed

        """
        print('saving embedding net...')
        torch.save(self.model.state_dict(), path_fname)


