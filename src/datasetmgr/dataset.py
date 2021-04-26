'''
Author: your name
Date: 2021-04-23 21:36:59
LastEditTime: 2021-04-23 21:36:58
LastEditors: your name
Description: In User Settings Edit
FilePath: /steganography_platform_pl/src/datasetmgr/dataset.py
'''
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.config import args
from icecream import ic

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

            # both cover and stego
            self.images = self.cover_path_list +  self.stego_path_list
            self.labels = np.concatenate([np.zeros(len(self.cover_path_list), dtype=np.int64), np.ones(len(self.stego_path_list), dtype=np.int64)])
            
            # only cover
            # self.images = self.cover_path_list
            # self.labels = np.zeros(len(self.cover_path_list), dtype=np.int64)

            # # only stego
            # self.images = self.stego_path_list
            # self.labels = np.ones(len(self.stego_path_list), dtype=np.int64)

            np.random.seed(seed+999)
            np.random.shuffle(self.images)
            np.random.seed(seed+999)
            np.random.shuffle(self.labels)

    def __getitem__(self, idx):
        if self.mode in ('train', 'val'):
            cover_img = np.array(Image.open(self.cover_path_list[idx]))
            stego_img = np.array(Image.open(self.stego_path_list[idx]))
            data = np.stack([cover_img, stego_img])
            
            if self.transforms:
                data = self.transforms(data)

            label = torch.tensor([0, 1], dtype=torch.int64)
        else:
            data = np.array(Image.open(self.images[idx]))
            if self.transforms:
                data = self.transforms(data)
            label = torch.tensor(self.labels[idx])
        return data, label

    def __len__(self):
        if self.mode in ('train', 'val'):
            return len(self.cover_path_list)
        else:
            return len(self.images)
