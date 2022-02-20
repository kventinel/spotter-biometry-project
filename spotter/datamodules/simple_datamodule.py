from pytorch_lightning import LightningDataModule


class SimpleDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
