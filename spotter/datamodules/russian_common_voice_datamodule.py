from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class RussianCommonVoiceDataset(Dataset):
    def __init__(self, data_dir: str, tsv_filename: str):
        data_path = Path(data_dir)
        tsv_path = data_path / tsv_filename
        info = pd.read_csv(tsv_path, sep='\t')
        self.filenames = list(map(lambda x: data_path / 'clips' / x, info['path']))
        self.text = list(info['sentence'])

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=10,
            n_fft=80,
            win_length=25,
            hop_length=25,
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        text = self.text[idx]
        if idx % 2000 == 0:
            print(text)
        waveform, sample_rate = torchaudio.load(filename)
        melspec = self.mel_spectrogram_transform(waveform)
        #print(melspec.shape)

        return melspec[:, :, :100]


class RussianCommonVoiceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_common_voice_dataset = RussianCommonVoiceDataset(
            data_dir=data_dir,
            tsv_filename='train.tsv',
        )
        self.val_common_voice_dataset = RussianCommonVoiceDataset(
            data_dir=data_dir,
            tsv_filename='dev.tsv',
        )
        self.test_common_voice_dataset = RussianCommonVoiceDataset(
            data_dir=data_dir,
            tsv_filename='test.tsv',
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_common_voice_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_common_voice_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_common_voice_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == '__main__':
    print('Datamodule checking...')

    data_dir = '/home/sergei/git/spotter-biometry-project/data/cv-corpus-8.0-2022-01-19/ru/'
    datamodule = RussianCommonVoiceDataModule(data_dir=data_dir)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    for i, batch in enumerate(train_dataloader):
        #print(batch)
        if i == 10:
            break
    for i, batch in enumerate(val_dataloader):
        #print(batch)
        if i == 10:
            break
    for i, batch in enumerate(test_dataloader):
        #print(batch)
        if i == 10:
            break

    print('Done!')
