#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2023/7/4 12:34
@License: (C) Copyright 2013-2023. 
@File: _embed_trainer.py
@Desc:

"""

import torch
from torch import optim
from ..utils import EmbDataset
from ..loss import vae_loss, ae_loss


class EmbNetTrainer:
    """

    Trainer for embedding networks. Used for pre-training or representation learning.

    """
    def __init__(self, model, beta=0.001, gamma=0.0001, c_max=25, batch_size=1024 * 2,
                 learning_rate=0.01, max_epochs=50, l2_weight=0.00001, lam=1e-4,
                 embedding='vae', model_type='vanilla_vae', device='cuda'):

        """

        pre-training or representation learning using embedding network

        Parameters
        ----------

        model
            Instance of `VAE` or `AE`, see `model._embed_net.VAE` or `model._embed_net.AE`
        beta
            weight used for vanilla VAE KL loss, default value = 0.001,
            see see `loss._loss.vae_loss`
        gamma
            weight used for beta VAE KL loss, default value = 0.0001
            see see `loss._loss.vae_loss`
        c_max
            maximum value of controllable value for beta VAE, default value = 25.0,
            see see `loss._loss.vae_loss`

        batch_size
            batch size during model training, default value = 1024 * 2
        learning_rate
            learning rate, default value = 0.01
        max_epochs
            the maximum number of epochs, default value = 50
        l2_weight
            weight factor for L2 regularization of model weights, default value = 0.00001
        lam
            weight factor for contractive loss, default value = 1e-4

        embedding
             Specifying embedding network. Could be `vae` or `ae`. Default value = 'vae'.

        model_type
            type of embedding networks.
                If `embedding = 'vae'`, then `model_type` could be `vanilla_vae` or `beta_vae`.
                If `embedding = 'ae'`, then `model_type` could be `vanilla_ae` or `contractive_ae`.
        device
            device used for the model
        """
        
        self.beta = beta
        self.gamma = gamma
        self.c_max = c_max

        self.device = device
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.embedding = embedding

        self.lam = lam

        self.model_type = model_type

        self.model = model.to(self.device)  # embed_net
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
        total_step = len(dataloader)
        for epoch in range(self.max_epochs):
            self.model.train()
            for i_step, (batch, ixs) in enumerate(dataloader):
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                if self.embedding == 'vae':

                    x_rec, z, mu, log_var, _ = self.model(batch)
                    train_loss = vae_loss(batch, x_rec, mu, log_var, self.max_epochs, epoch, self.device,
                                          beta=self.beta, gamma=self.gamma, c_max=self.c_max,
                                          model_type=self.model_type)
                if self.embedding == 'ae':
                    x_rec, z, _ = self.model(batch)

                    w = self.model.z_layer[0].weight
                    train_loss = ae_loss(batch, x_rec, model_type=self.model_type, w=w, h=z, lam=self.lam)

                loss = train_loss['total_loss']
                loss.backward()
                self.optimizer.step()

                if (i_step + 1) % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], total Loss: {:.4f}, recons Loss: {:.4f}, KL: {:.4f}'
                          .format(epoch + 1, self.max_epochs, i_step + 1, total_step,
                                  loss.item(), train_loss['recons_loss'].item(),
                                  train_loss['KL_emb_loss'].item() if self.embedding == 'vae' else 0))

    def predict(self, x):
        """

        representation learning

        Parameters
        ----------
        x
            raw input 2d np.array with shape (N_sample, N_feature)

        Returns
        -------
        z_new.detach().cpu().numpy()
            sample representation learned by the embedding network

        """
        x = torch.tensor(x)
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            if self.embedding == 'vae':
                x_rec, z_new, mu, log_var, _ = self.model(x)
            if self.embedding == 'ae':
                x_rec, z_new, _ = self.model(x)
            return z_new.detach().cpu().numpy()

    def save_model(self, path_fname):
        """

        Parameters
        ----------
        path_fname
            file used for saving the model

        """
        print('saving embedding net...')
        torch.save(self.model.state_dict(), path_fname)