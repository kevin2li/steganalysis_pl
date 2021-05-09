#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
from src.utils import ABS, HPF, TLU, SPPLayer

__all__ = ['YeNet']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

srm_16filters = np.load('src/utils/srm_16_filters.npy')
srm_minmax = np.load('src/utils/minmax_filters.npy')
srm_filters = np.concatenate((srm_16filters, srm_minmax),axis=0)
srm_filters = torch.from_numpy(srm_filters).to(device=device, dtype=torch.float32)

class YeNet(pl.LightningModule):
    def __init__(self, lr: float=0.001, weight_decay: float= 5e-4, gamma: float = 0.2, momentum: float = 0.9, step_size: int = 50, **kwargs):
        super(YeNet, self).__init__()
        # 超参
        # for optimizer(SGD)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        # for lr scheduler(StepLR)
        self.gamma = gamma
        self.step_size = step_size
        # 其他
        self.save_hyperparameters()
        self.accuracy = Accuracy()

        # 组网
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 30, 5, 1),
            TLU(3.0),
        )
        self.layer1[0].weight = nn.Parameter(srm_filters, requires_grad=False)

        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 30, 3, 1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(30, 30, 3, 1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(30, 30, 3, 1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(30, 32, 5, 1),
            nn.ReLU(),
            nn.AvgPool2d(3, 2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1),
            nn.ReLU(),
            nn.AvgPool2d(3, 2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1),
            nn.ReLU(),
            nn.AvgPool2d(3, 2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1),
            nn.ReLU(),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 3),
            nn.ReLU(),
        )
        self.layer10 = nn.Sequential(
            nn.Linear(3*3*16, 2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out1 = self.layer2(out)
        out2 = self.layer3(out1)
        out3 = self.layer4(out2)
        out4 = self.layer5(out3)
        out5 = self.layer6(out4)
        out6 = self.layer7(out5)
        out7 = self.layer8(out6)
        out8 = self.layer9(out7)
        out8 = out8.reshape(out8.shape[0], -1)
        out9 = self.layer10(out8)
        return out9

    def configure_optimizers(self):
        params_wd, params_rest = [], []
        for m in self.parameters():
            if m.requires_grad:
                (params_wd if m.dim()!=1 else params_rest).append(m)
        param_groups = [{'params': params_wd, 'weight_decay': self.weight_decay},
                        {'params': params_rest}]
        optimizer = torch.optim.Adadelta(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        train_loss = F.cross_entropy(y_hat, y)
        y_hat = F.softmax(y_hat, dim=1)
        train_acc = self.accuracy(y_hat, y)
        # record
        lr = [group['lr'] for group in self.optimizers().param_groups]

        self.log('train_loss', train_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('lr', lr, prog_bar=True, on_step=False, on_epoch=True)
        # self.logger.experiment.add_image('example_images', x[0], 0)
        return {'loss':train_loss, 'train_loss': train_loss, 'train_acc': train_acc}

    def validation_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        y_hat = F.softmax(y_hat, dim=1)
        val_acc = self.accuracy(y_hat, y)
        # record
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return {'val_loss': val_loss, 'val_acc': val_acc}
    
    def test_step(self, batch, batch_idx):
        # preprocess
        x, y = batch
        N, C, H, W = x.shape
        x = x.reshape(N*C, 1, H, W)
        y = y.reshape(-1)
        # forward
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        y_hat = F.softmax(y_hat, dim=1)
        test_acc = self.accuracy(y_hat, y)
        # record
        self.log('test_loss', test_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', test_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return {'test_loss': test_loss, 'test_acc': test_acc}
