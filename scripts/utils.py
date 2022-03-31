from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import tqdm
from torchaudio.sox_effects import apply_effects_tensor

def save_melspecs(filenames, out_path):
    print('Saving melspecs...')

    for filename in tqdm.tqdm(filenames):       
        wave_form, sample_rate = torchaudio.load(filename, normalize=True)
        
        effects = [
            ['gain', '-n'],  # normalises to 0dB
        ]   
        waveform, sample_rate = apply_effects_tensor(
            wave_form, sample_rate, effects)  
        
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
        )
        melspec = mel_spectrogram_transform(waveform)
        min_val = 1e-5
        log_melspec = torch.log(torch.clamp(melspec, min_val))
        torch.save(log_melspec, out_path / (filename.stem + '.pt'))

    print('Done!')
