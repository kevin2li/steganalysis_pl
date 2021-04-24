# %%
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from .dataset import ImageDataset

__all__ = ['ImageDataModule']

class AugData():
    def __call__(self, data):
        # Rotation
        rot = np.random.randint(0, 3)
        data = np.rot90(data, rot, axes=[1, 2]).copy()

        # Mirroring
        if np.random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        return data

class ToTensor():
    def __call__(self, data):
        data = data.astype(np.float32)
        # data = np.expand_dims(data, 1)
        data = data / 255.0
        return torch.from_numpy(data)

train_transform = T.Compose([
    AugData(),
    ToTensor(),
])

eval_transform = T.Compose([
    ToTensor()
])

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs = ['data/bb_wow_0.4', 'data/cover'], batch_size: int = 32, train_transform=train_transform, eval_transform=eval_transform):
        super(ImageDataModule, self).__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(dataset_dir=self.data_dirs, mode='train', transforms=self.train_transform)
        self.val_dataset = ImageDataset(dataset_dir=self.data_dirs, mode='val', transforms=self.eval_transform)
        self.test_dataset = ImageDataset(dataset_dir=self.data_dirs, mode='test', transforms=self.eval_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)
