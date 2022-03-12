from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import tqdm


def save_melspecs(filenames, out_path):
    print('Saving melspecs...')

    for filename in tqdm.tqdm(filenames):
        waveform, sample_rate = torchaudio.load(filename)
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
        )
        melspec = mel_spectrogram_transform(waveform)
        min_val = 1e-5
        log_melspec = torch.log(torch.clamp(melspec, min_val))
        torch.save(log_melspec, out_path / (filename.stem + '.pt'))

    print('Done!')
