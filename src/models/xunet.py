#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.utils import ABS, HPF, TLU, SPPLayer
from torchmetrics import Accuracy

__all__ = ['XuNet']

class XuNet(pl.LightningModule):
    def __init__(self, lr: float=0.001, weight_decay: float= 5e-4, gamma: float = 0.2, momentum: float = 0.9, step_size: int = 200, **kwargs):
        super(XuNet, self).__init__()
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
        self.KV = nn.Conv2d(1, 1, 5, padding=2)
        KV = torch.tensor([[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.
        KV = KV.reshape(1, 1, 5, 5)
        self.KV.weight = nn.Parameter(KV, requires_grad=False)

        self.group1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=1, padding=2, bias=False),
            ABS(),
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.AvgPool2d(5, 2)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(5, 2)
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(16, 32, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(5, 2)
        )
        self.group4 = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(5, 2)
        )
        self.group5 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 2),
        )

    def forward(self, x):
        out = self.KV(x)
        out1 = self.group1(out)
        out2 = self.group2(out1)
        out3 = self.group3(out2)
        out4 = self.group4(out3)
        out5 = self.group5(out4)
        out5 = out5.reshape(out5.shape[0], -1)
        out6 = self.fc(out5)
        return out6

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.gamma, patience=self.patience, cooldown=self.cooldown)
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
