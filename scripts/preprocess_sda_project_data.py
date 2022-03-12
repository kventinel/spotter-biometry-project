from pathlib import Path

import pandas as pd

from utils import save_melspecs


def save_new_tsv(info, new_tsv_path) -> None:
    info['mel_name'] = info['name']
    info['mel_name'] = info['mel_name'].apply(lambda x: x.replace('wav', 'pt'))
    info.to_csv(new_tsv_path, sep='\t')


def preprocess_sda_project_data(data_dir, out_dir) -> None:
    tsv_filename = f'data.tsv'
    new_tsv_filename = f'data_with_melspecs.tsv'

    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tsv_path = data_path / tsv_filename
    new_tsv_path = data_path / new_tsv_filename

    info = pd.read_csv(tsv_path, sep='\t')
    filenames = \
        list(map(lambda x: data_path / 'audio' / x, info['name']))
    save_melspecs(filenames=filenames, out_path=out_path)
    save_new_tsv(info, new_tsv_path=new_tsv_path)


def main():
    data_dir = '/home/sergei/git/spotter-biometry-project/data/sda-project-set'
    out_dir = f'{data_dir}/melspecs'
    preprocess_sda_project_data(data_dir=data_dir, out_dir=out_dir)
    print('Done!')


if __name__ == '__main__':
    main()