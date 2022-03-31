#import hydra
#from omegaconf import DictConfig, OmegaConf
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

#from spotter.configs import *
from spotter.datamodules import RussianCommonVoiceDataModule, SDAProjectDataModule
from spotter.models import SimpleClassifier


#@hydra.main(config_path='./configs', config_name='defaults')
def main(args):
    print('Start...')
    model = SimpleClassifier()#.to(args['device'])
    print(model)
    datamodule = SDAProjectDataModule(
        data_dir=args['data_dir'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
    )

    if args['dev']:
        logger = None
        overfit_batches = 10
    else:
        logger = WandbLogger(project="spotter-biometry")
        overfit_batches = 0

    trainer = Trainer(
        gpus=1,
        logger=logger,
        overfit_batches=overfit_batches,
    )
    trainer.fit(model, datamodule=datamodule)
    print('Done!')


if __name__ == '__main__':
    args = {
        'batch_size': 64,
        'data_dir': '/home/sergei/git/spotter-biometry-project/data/sda-project-set',
        'dev': 0,
        'device': torch.device('cuda'),
        'num_workers': 8,
    }
    main(args)

