import torch
import torch.nn as nn
import torchmetrics
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule


class SimpleClassifier(LightningModule):
    def __init__(
        self,
        embedding_size: int = 100,
        window_size: int = 40,
        n_keywords: int = 1,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.n_keywords = n_keywords

        self.embedder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size),
        )
        self.head = nn.Sequential(
            Rearrange('bs w e -> bs (w e)'),
            nn.Linear(window_size * embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_keywords + 1),
            nn.Softmax(),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        columns = torch.split(x, [1 for i in range(x.shape[2])], dim=2)
        columns = list(map(lambda x: x.squeeze(2), columns))
        embeddings = list()

        for column in columns:
            embedding = self.embedder(column).unsqueeze(1)
            embeddings.append(embedding)

        probs = list()

        for i in range(len(embeddings) - self.window_size):
            x = torch.cat(embeddings[i:i+self.window_size], dim=1)#.unsqueeze(0)
            prob = self.head(x)
            probs.append(prob)

        return probs

    def training_step(self, batch, batch_idx):
        melspecs = batch['melspecs']
        labels = batch['labels']
        probs = self.forward(melspecs)
        #print('!', len(probs), probs[0].shape, labels.shape)

        loss = 0
        for i in range(len(probs)):
            loss += self.criterion(probs[i], labels)
        loss /= len(probs)
        accuracy = torchmetrics.functional.accuracy(probs[len(probs) // 2], labels)

        if batch_idx % 100 == 0:
            print('-----------')
            print(loss)
            print(accuracy)
            print(labels)

        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)

        return {
            'loss': loss,
            'accuracy': accuracy,
        }

    def validation_step(self, batch, batch_idx):
        melspecs = batch['melspecs']
        labels = batch['labels']
        probs = self.forward(melspecs)
        #print('!', len(probs), probs[0].shape, labels.shape)

        loss = 0

        for i in range(len(probs)):
            loss += self.criterion(probs[i], labels)

        loss /= len(probs)

        accuracy = torchmetrics.functional.accuracy(probs[len(probs) // 2], labels)

        if batch_idx % 100 == 0:
            print('-----------')
            print(loss)
            print(accuracy)
            print(labels)

        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)

        return {
            'loss': loss,
            'accuracy': accuracy,
        }

    def _update_memory(self, x) -> None:
        '''updates self.embeddings'''
        self.embeddings.pop(0)
        current_embedding = self.embedder.forward(x)
        self.embeddings.append(current_embedding)

    def streaming(self, audio_sequence):
        '''returns signals; signal - what a model thinks about the keyword said at i-th column'''
        self.embeddings = [0 for i in range(self.window_size)]

        for i in range(self.window_size):
            current_embedding = self.embedder(audio_sequence[i])
            self.embeddings.append(current_embedding)

        signals = list()

        for i in range(self.window_size, len(audio_sequence)):
            x = torch.cat(self.embeddings).unsqueeze(0)
            signal = self.head(x)
            signals.append(signal)
            self._update_memory(audio_sequence[i])

        return signals

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


if __name__ == '__main__':
    model = SimpleClassifier()
    print(model)
