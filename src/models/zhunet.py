'''
Author: your name
Date: 2021-04-23 20:38:55
LastEditTime: 2021-04-23 21:15:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /steganography_platform_pl/src/models/zhunet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.utils import ABS, SPPLayer, HPF, TLU
from icecream import ic
import numpy as np
import pytorch_lightning as pl
__all__ = ['ZhuNet']



class ZhuNet(pl.LightningModule):
    def __init__(self, lr: float=0.005, weight_decay: float= 5e-4, milestones=[40, 80, 120], gamma: float = 0.2, momentum: float = 0.9):
        super(ZhuNet, self).__init__()
        # 超参
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma
        self.momentum = momentum
        self.save_hyperparameters()
        self.accuracy = pl.metrics.Accuracy()

        # 组网
        self.layer1 = HPF()
        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1, groups=30),
            ABS(),
            nn.Conv2d(60, 30, kernel_size=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1, groups=30),
            nn.Conv2d(60, 30, kernel_size=1),
            nn.BatchNorm2d(30)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer8 = SPPLayer(3)
        self.fc = nn.Sequential(
            nn.Linear(128*21, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        
        # sepconv
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = out3 + out1

        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.fc(out8)

        return out9
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=25, cooldown=5)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        train_loss = F.cross_entropy(y_hat, y)
        train_acc = self.accuracy(y_hat, y)
        # record
        lr = [group['lr'] for group in self.optimizers().param_groups]

        self.log('train_loss', train_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('lr', lr, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = self.accuracy(y_hat, y)
        # record
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss
    
    def test_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        test_acc = self.accuracy(y_hat, y)
        # record
        self.log('test_acc', test_acc, prog_bar=True, on_step=False, on_epoch=True)
        return test_acc