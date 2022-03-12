from pathlib import Path

import pandas as pd

from utils import save_melspecs


def save_new_tsv(info, new_tsv_path) -> None:
    info['mel_path'] = info['path']
    info['mel_path'] = info['mel_path'].apply(lambda x: x.replace('mp3', 'pt'))
    info.to_csv(new_tsv_path, sep='\t')


def preprocess_common_voice_data(data_dir, out_dir, mode='train') -> None:
    tsv_filename = f'{mode}.tsv'
    new_tsv_filename = f'{mode}_with_melspecs.tsv'

    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tsv_path = data_path / tsv_filename
    new_tsv_path = data_path / new_tsv_filename

    info = pd.read_csv(tsv_path, sep='\t')
    filenames = \
        list(map(lambda x: data_path / 'clips' / x, info['path']))
    save_melspecs(filenames=filenames, out_path=out_path)
    save_new_tsv(info, new_tsv_path=new_tsv_path)


def main():
    data_dir = '/home/sergei/git/spotter-biometry-project/data/cv-corpus-8.0-2022-01-19/ru'
    out_dir = f'{data_dir}/melspecs'
    preprocess_common_voice_data(data_dir=data_dir, out_dir=out_dir, mode='train')
    preprocess_common_voice_data(data_dir=data_dir, out_dir=out_dir, mode='dev')
    preprocess_common_voice_data(data_dir=data_dir, out_dir=out_dir, mode='test')
    print('Done!')


if __name__ == '__main__':
    main()

