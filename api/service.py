import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as FA
import torchaudio.transforms as TA
import torchlibrosa as TL

from model import BiometryModel, ConvEncoder, Discriminator
from config import *


class BiometryService:
    def __init__(self, checkpoint_path, model_treshold, resample_rate=32000, waveform_treshold=200000):
        self.model = BiometryModel.load_from_checkpoint(
            checkpoint_path,
            encoder=ConvEncoder(in_time=626, in_features=128, embedding_size=256),
            discriminator=Discriminator(input_size=256, d_size=200, hidden_size=256,
                                        negative_slope=0.2, use_cos=True, use_net=True)
        )
        self.model.eval()
        self.resample_rate = resample_rate
        self.waveform_treshold = waveform_treshold
        self.transforms = nn.Sequential(
            TL.Spectrogram(
                n_fft=1024,
                hop_length=320,
            ), TL.LogmelFilterBank(
                sr=self.resample_rate,
                n_mels=128,
                n_fft=1024,
                fmin=20,
                fmax=16000,
            )
        )
        self.model_treshold = model_treshold

    def preprocessing(self, filename):
        waveform, sample_rate = torchaudio.load(filename, normalize=True)

        resampled_waveform = FA.resample(waveform,
                                         sample_rate,
                                         self.resample_rate)

        if resampled_waveform.shape[1] >= self.waveform_treshold:
            resampled_waveform = resampled_waveform[:, :self.waveform_treshold]
        else:
            resampled_waveform = F.pad(
                resampled_waveform,
                (0, self.waveform_treshold - resampled_waveform.shape[1], 0, 0),
                value=resampled_waveform.min()
            )

        if self.transforms is not None:
            resampled_waveform = self.transforms(resampled_waveform)

        return resampled_waveform

    def set_specimen_d_vector(self):
        spec_records_paths = glob.glob(f'{os.path.join(DIRECTORY_SAMPELS, TEMPLATE_NAME_EXAMPLES_VOICE.format("*"))}')
        spec_X = torch.cat([self.preprocessing(spec_records_path) for spec_records_path in spec_records_paths], dim=0)

        with torch.no_grad():
            self.model.set_specimen_d_vector(spec_X)

    def predict(self):
        x = self.preprocessing(os.path.join(DIRECTORY_SAMPELS, TEMPLATE_REC_VOICE))

        with torch.no_grad():
            return int(self.model(x) > self.model_treshold)
