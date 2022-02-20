from pytorch_lightning import Trainer

#from spotter.configs import *
from spotter.datamodules import RussianCommonVoiceDataModule
from spotter.models import SimpleModel


def main(args):
    print('Start...')
    model = SimpleModel()
    print(model)
    datamodule = RussianCommonVoiceDataModule()
    trainer = Trainer()
    trainer.fit(model, datamodule=datamodule)
    print('Done!')


if __name__ == '__main__':
    #main(args)
    main(None)

