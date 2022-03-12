#import hydra
#from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

#from spotter.configs import *
from spotter.datamodules import RussianCommonVoiceDataModule, SDAProjectDataModule
from spotter.models import SimpleClassifier


#@hydra.main(config_path='./configs', config_name='defaults')
def main(args):
    print('Start...')
    model = SimpleClassifier()
    print(model)
    data_dir = '/home/sergei/git/spotter-biometry-project/data/sda-project-set'
    datamodule = SDAProjectDataModule(data_dir=data_dir)

    if args['dev']:
        logger = None
    else:
        logger = WandbLogger(project="spotter-biometry")

    trainer = Trainer(logger=logger)
    trainer.fit(model, datamodule=datamodule)
    print('Done!')


if __name__ == '__main__':
    args = {
        'dev': True,
    }
    main(args)

