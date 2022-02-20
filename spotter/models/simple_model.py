import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 3))
        self.decoder = nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 28 * 28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
