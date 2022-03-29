import librosa
import numpy as np
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import tqdm
import pickle
from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as FA
import torchaudio
import torchaudio.transforms as TA
import torchlibrosa as TL
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from torchaudio.sox_effects import apply_effects_tensor

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    
from spotter.models.MatchBox import MatchBox
    
with open('../spotter/models/id2number.pkl', 'rb') as f:
    id2number = pickle.load(f)
    
with open('../spotter/models/number2id.pkl', 'rb') as f:
    number2id = pickle.load(f)

UNK = "UNK"
UNK_CLASS = 0
assert id2number[UNK] == UNK_CLASS 
assert number2id[UNK_CLASS] == UNK    
    
MODEL_PATH = '../spotter/models/MatchBox_3_1_64_Vol.pth'
N_CLASSES = len(id2number)    



import wave
import pyaudio as pa

class _Keyword_Spotting_Service:
    
    _instance = None
    
    def __init__(self,  
                 resample_rate = 16000
                ):
        super().__init__()
        self._model = MatchBox(n_classes = N_CLASSES)
        self._model.load_state_dict(torch.load(MODEL_PATH))
        self._model.eval() 
        self._id2number = copy(id2number)
        self._number2id = copy(number2id)
        
        self._resample_rate = resample_rate

        
    
    def predict(self, file_path):
        melspec = self.preprocess(file_path)
        # make prediction   
        predictions = self._model(melspec)
        predicted_index = predictions.argmax(dim=1).item()
        predicted_keyword = self._number2id[predicted_index]
        return predicted_keyword

    def preprocess(self, filename, n_fft=1024):
        # load audio file
        wave_form, sample_rate = torchaudio.load(filename, normalize=True)
             
        effects = [
            ['gain', '-n'],  # normalises to 0dB
        ]   
        waveform, sample_rate = apply_effects_tensor(
            wave_form, sample_rate, effects)  
                                
        mel_spectrogram_transform = nn.Sequential(
            T.Resample(sample_rate, self._resample_rate),
            T.MelSpectrogram(
                sample_rate=self._resample_rate,
                n_fft=n_fft)
        )
        melspec = mel_spectrogram_transform(waveform)
        min_val = 1e-5
        log_melspec = torch.log(torch.clamp(melspec, min_val))
        return log_melspec 

def Keyword_Spotting_Service():
    # ensure that we only have 1 instance
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service() 
    return _Keyword_Spotting_Service._instance


if __name__ == '__main__':
    kss = Keyword_Spotting_Service()
    keyword1 = kss.predict('tmp/Light.wav')
    print(keyword1)