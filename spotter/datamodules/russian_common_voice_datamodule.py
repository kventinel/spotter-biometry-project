import string
from collections import Counter
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
        self.filenames = \
            list(map(lambda x: data_path / 'clips' / x, info['path']))
        self.texts = list(info['sentence'])
        prepared_texts = list(map(
            lambda x: "".join(
                symbol for symbol in x if symbol not in string.punctuation
            ).lower(),
            info['sentence'],
        ))
        splitted_texts = list(map(lambda x: x.split(), prepared_texts))
        words_counter = self._get_counter(splitted_texts)
        print(words_counter.most_common()[:20])

        keywords = ['должны', 'слово', 'сейчас']
        self.labels = self._get_labels(splitted_texts, keywords)

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=10,
            n_fft=80,
            win_length=25,
            hop_length=25,
        )

    def _get_counter(self, splitted_texts):
        words_counter = Counter()

        for words in splitted_texts:
            for word in words:
                if len(word) < 4:
                    continue
                words_counter[word] += 1

        return words_counter

    def _get_labels(self, splitted_texts, keywords):
        labels = list(0 for _ in range(len(splitted_texts)))

        for i, text in enumerate(splitted_texts):
            for j, keyword in enumerate(keywords):
                if keyword in text:
                    labels[i] = j+1

        return labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        #text = self.texts[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(filename)
        #melspec = self.mel_spectrogram_transform(waveform)
        #print(melspec.shape)

        return waveform, label


class Collator:
    def __call__(self, batch):
        waveforms, labels = zip(*batch)

        lengths = list()
        for waveform in waveforms:
            lengths.append(waveform.size(-1))

        batch_waveforms = torch.zeros(len(batch), max(lengths))
        for i, (waveform, length) in enumerate(zip(waveforms, lengths)):
            batch_waveforms[i, :length] = waveform.squeeze()

        labels = torch.tensor(labels).long()
        lengths = torch.tensor(lengths).long()

        # return {
        #     'wav': batch_waveforms,
        #     'label': labels,
        #     'length': lengths,
        # }

        return batch_waveforms, labels, lengths


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
            data_dir=self.data_dir,
            tsv_filename='train.tsv',
        )
        self.val_common_voice_dataset = RussianCommonVoiceDataset(
            data_dir=self.data_dir,
            tsv_filename='dev.tsv',
        )
        self.test_common_voice_dataset = RussianCommonVoiceDataset(
            data_dir=self.data_dir,
            tsv_filename='test.tsv',
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_common_voice_dataset,
            batch_size=self.batch_size,
            collate_fn=Collator(),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_common_voice_dataset,
            batch_size=self.batch_size,
            collate_fn=Collator(),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_common_voice_dataset,
            batch_size=self.batch_size,
            collate_fn=Collator(),
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
        print(batch)
        if i == 10:
            break
    for i, batch in enumerate(val_dataloader):
        if i == 10:
            break
    for i, batch in enumerate(test_dataloader):
        if i == 10:
            break

    print('Done!')
