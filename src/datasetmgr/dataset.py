import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ['ImageDataset']

class ImageDataset(Dataset):
    def __init__(self, data_dirs, transforms=None, mode='train', seed: int = 2021):
        super(ImageDataset, self).__init__()
        self.data_dirs = data_dirs
        self.mode = mode
        self.transforms = transforms
        self.stego_path_list = []
        for _ in data_dirs[:-1]:
            self.stego_path_list.extend(glob.glob(os.path.join(_,'*.png')))
        self.cover_path_list = glob.glob(os.path.join(data_dirs[-1],'*.png'))
        np.random.seed(seed)
        np.random.shuffle(self.cover_path_list)
        np.random.seed(seed)
        np.random.shuffle(self.stego_path_list)
        if self.mode == 'train':
            self.cover_path_list = self.cover_path_list[:4000]
            self.stego_path_list = self.stego_path_list[:4000]
        elif self.mode == 'val':
            self.cover_path_list = self.cover_path_list[4000:5000]
            self.stego_path_list = self.stego_path_list[4000:5000]
        else:
            self.cover_path_list = self.cover_path_list[5000:]
            self.stego_path_list = self.stego_path_list[5000:]

    def __getitem__(self, idx):
        cover_img = np.array(Image.open(self.cover_path_list[idx]))
        stego_img = np.array(Image.open(self.stego_path_list[idx]))
        data = np.stack([cover_img, stego_img])
        
        if self.transforms:
            data = self.transforms(data)

        label = torch.tensor([0, 1], dtype=torch.int64)

        return data, label

    def __len__(self):
        return len(self.cover_path_list)
