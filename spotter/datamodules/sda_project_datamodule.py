import string
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import tqdm
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)


class SDAProjectDataset(Dataset):
    def __init__(self, data_dir: str):
        data_path = Path(data_dir)
        tsv_path = data_path / 'data_with_melspecs.tsv'
        info = pd.read_csv(tsv_path, sep='\t')
        self.paths = \
            list(map(lambda x: data_path / 'melspecs' / x, info['mel_name']))
        texts = list(info['text'])
        prepared_texts = list(map(
            lambda x: "".join(
                symbol for symbol in x if symbol not in string.punctuation
            ).lower(),
            texts,
        ))
        splitted_texts = list(map(lambda x: x.split(), prepared_texts))
        keywords = ['громкость']
        self.labels = self._get_labels(splitted_texts, keywords)

    def _get_labels(self, splitted_texts, keywords):
        labels = list(0 for _ in range(len(splitted_texts)))

        for i, text in enumerate(splitted_texts):
            for j, keyword in enumerate(keywords):
                if keyword in text:
                    labels[i] = j + 1

        return np.asarray(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        melspec = torch.load(path)

        return melspec, label


class Collator:
    def __call__(self, batch):
        melspecs, labels = zip(*batch)

        lengths = list()
        for melspec in melspecs:
            lengths.append(melspec.size(-1))

        # -11.52 is torch.log(1e-5) - padding for zero noise
        batch_melspecs = -11.52 * torch.ones(
            len(batch),
            melspecs[0].shape[1],
            max(lengths),
        )
        for i, (melspec, length) in enumerate(zip(melspecs, lengths)):
            batch_melspecs[i, :, :length] = melspec.squeeze()

        labels = torch.tensor(labels).long()
        lengths = torch.tensor(lengths).long()

        return {
            'melspecs': batch_melspecs,
            'labels': labels,
            'lengths': lengths,
        }


class SDAProjectDataModule(LightningDataModule):
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

    def setup(self, stage: Optional[str] = None, ratios=(0.8, 0.1, 0.1)):
        full_dataset = SDAProjectDataset(self.data_dir)
        test_dataset_length = int(ratios[2] * len(full_dataset))
        val_dataset_length = int(ratios[1] * len(full_dataset))
        train_dataset_length = \
            len(full_dataset) - test_dataset_length - val_dataset_length

        full_indices = np.arange(len(full_dataset))
        trainval_indices, test_indices = train_test_split(
            full_indices,
            test_size=ratios[2],
            stratify=full_dataset.labels,
        )
        train_indices, val_indices = train_test_split(
            trainval_indices,
            test_size=ratios[1],
            stratify=full_dataset.labels[trainval_indices],
        )

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)
        self.train_dataset.labels = full_dataset.labels[train_indices]
        self.val_dataset.labels = full_dataset.labels[val_indices]
        self.test_dataset.labels = full_dataset.labels[test_indices]

        self.train_sampler = self._get_sampler(self.train_dataset)
        self.val_sampler = self._get_sampler(self.val_dataset)
        self.test_sampler = self._get_sampler(self.test_dataset)

    def _get_sampler(self, dataset):
        labels =  np.array(dataset.labels)
        pos = sum(labels)
        neg = len(labels) - pos
        weights = torch.tensor([1 / pos if t == 1 else 1 / neg for t in labels])
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=Collator(),
            num_workers=self.num_workers,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=Collator(),
            num_workers=self.num_workers,
            sampler=self.val_sampler,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=Collator(),
            num_workers=self.num_workers,
            sampler=self.test_sampler,
        )


if __name__ == '__main__':
    data_dir = '/home/sergei/git/spotter-biometry-project/data/sda-project-set'
    datamodule = SDAProjectDataModule(data_dir=data_dir)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    a = 0

    for i, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        labels = batch['labels']
        if i < 10:
            print(labels)
        a += labels.sum()

    print(a, a / len(train_dataloader) / 8, len(train_dataloader) * 8)
    for i, batch in enumerate(val_dataloader):
        if i == 10:
            break
    for i, batch in enumerate(test_dataloader):
        if i == 10:
            break

    print('Done!')
