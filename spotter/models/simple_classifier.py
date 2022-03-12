import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class SimpleClassifier(LightningModule):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            nn.AdaptiveMaxPool1d(output_size=n_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        print('!!!!', x.shape)
        probs = self.body(x)
        probs = probs.squeeze(1)
        return probs

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        melspecs = batch['melspecs']
        labels = batch['labels']
        outputs = self.forward(melspecs)
        print('!', outputs.shape, labels.shape)
        loss = self.criterion(outputs, labels)
        return loss

    def _update_memory(self, memory, x):
        pass

    def streaming(self, audio_sequence):
        window = audio_sequence[:self.window_size]
        memory = list()

        for i in range(self.window_size, len(audio_sequence)):
            #single_feature = self.
            self._update_memory(memory, audio_sequence[i])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


if __name__ == '__main__':
    model = SimpleClassifier()
    print(model)
