from torch import Generator
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):

    def __init__(self, batch_size: int, dataloader_num_workers: int, random_seed: int):
        super(BaseDataModule, self).__init__()

        self.training_set = None
        self.validation_set = None
        self.test_set = None

        self.generator = Generator()
        if random_seed is not None:
            self.generator.manual_seed(random_seed)

        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.random_seed = random_seed

    def train_dataloader(self):
        assert self.training_set is not None
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.dataloader_num_workers, generator=self.generator)

    def val_dataloader(self):
        assert self.validation_set is not None
        return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        assert self.test_set is not None
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.dataloader_num_workers)