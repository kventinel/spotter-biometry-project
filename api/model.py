import os
import pandas as pd
import numpy as np

import librosa
import wave

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as FA
import torchaudio
import torchaudio.transforms as TA
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size):
        super().__init__()

        self.dense = nn.Linear(input_size, output_size)

    def forward(self, X):
        return self.dense(X.mean(dim=1))


class Discriminator(nn.Module):
    def __init__(self,
                 input_size=256,
                 d_size=200,
                 hidden_size=256,
                 negative_slope=0.2,
                 use_cos=True,
                 use_net=True):
        super().__init__()
        assert use_cos or use_net

        self.cos_disc = nn.CosineSimilarity(dim=1)
        self.d_size = d_size

        self.net_disc = nn.Sequential(
            nn.Linear(2 * input_size, hidden_size),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_size, 1)
        )

        self.use_cos = use_cos
        self.use_net = use_net
        self.score_combinator = nn.Linear(2, 1)

    def forward(self, X, specimen_X):
        if self.use_cos:
            cos_output = self.cos_disc(X[:, :self.d_size], specimen_X[:, :self.d_size])[..., None]

        if self.use_net:
            specimen_X_rep = specimen_X.repeat(X.shape[0], 1)
            net_output = self.net_disc(torch.cat([X, specimen_X_rep], dim=1))

        if self.use_cos and self.use_net:
            output = torch.cat([cos_output, net_output], dim=1)
            return self.score_combinator(output)
        elif self.use_cos:
            return cos_output
        elif self.use_net:
            return net_output


def GE2E_XS_Loss(scoring_block):
    scoring_block = scoring_block - scoring_block.max()
    exp_scoring_block = torch.exp(scoring_block)

    diag = torch.diagonal(scoring_block)
    exp_diag = torch.diagonal(exp_scoring_block)
    exp_block_sum = exp_scoring_block.sum()
    exp_diag_sum = exp_diag.sum()

    loss = -torch.sum(diag - torch.log(exp_diag + exp_block_sum - exp_diag_sum))

    return loss


class SimpleModel(pl.LightningModule):
    def __init__(self, use_cos=True, use_net=True):
        super().__init__()

        self._encoder = Encoder(
            input_size=128,
            output_size=256
        )

        self._discriminator = Discriminator(use_cos=use_cos, use_net=use_net)
        self._loss = GE2E_XS_Loss

        self._tmp_i = 0

        self._specimen_d_vector = None

    def set_specimen_d_vector(self, x):
        encod_sounds = self._encoder(x)
        self._specimen_d_vector = torch.mean(encod_sounds, dim=0, keepdims=True)

    def get_specimen_d_vector(self):
        return self._specimen_d_vector

    def forward(self, x):
        assert self._specimen_d_vector != None, "The voice sample vector is not set"

        encod_sounds = self._encoder(x)

        return self._discriminator(encod_sounds, self._specimen_d_vector)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X_batch, specimen_X_batch, _ = train_batch

        loss = 0

        n_speakers, n_records_per_speaker, n_times, n_features = X_batch.shape
        n_specimens = specimen_X_batch.shape[1]

        specimen_X = specimen_X_batch.view(n_speakers * n_specimens, n_times, n_features)
        encod_specimen_X = self._encoder(specimen_X).view(n_speakers, n_specimens, -1)

        X = X_batch.view(n_speakers * n_records_per_speaker, n_times, n_features)
        encod_X = self._encoder(X)


        list_speakers_dist = []

        for specimen_x in encod_specimen_X:
            specimen_x = torch.mean(specimen_x, dim=0, keepdims=True)
            list_speakers_dist.append(self._discriminator(encod_X, specimen_x))

        speakers_dist = torch.concat(list_speakers_dist, dim=1)
        speakers_dist = speakers_dist.view(n_speakers, n_records_per_speaker, n_speakers)
        score_tensor = speakers_dist.permute(1, 0, 2)

        for score_block in score_tensor:
            l = self._loss(score_block)
            loss += l

        loss /= n_speakers

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        pass