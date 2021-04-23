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
from src.utils import ABS, SPPLayer, HPF, TLU
from icecream import ic
import numpy as np
import pytorch_lightning as pl
__all__ = ['ZhuNet']



class ZhuNet(pl.LightningModule):
    def __init__(self, lr: float=0.005, weight_decay: float= 5e-4):
        super(ZhuNet, self).__init__()
        # 超参
        self.lr = lr
        self.weight_decay = weight_decay
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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        # record
        self.log('batch_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        # record
        self.log('batch_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        acc = self.accuracy(y_hat, y)
        # record
        self.log('acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return acc