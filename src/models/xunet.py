#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__all__ = ['XuNet']

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

KV = torch.tensor([[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]) / 12.
KV = KV.view(1, 1, 5, 5).to(device=device, dtype=torch.float)
KV = torch.autograd.Variable(KV, requires_grad=False)

class XuNet(pl.LightningModule):
    def __init__(self, lr: float=0.005, weight_decay: float= 5e-4, gamma: float = 0.2, momentum: float = 0.9, patience: int = 20, cooldown: int = 5, **kwargs):
        super(XuNet, self).__init__()
        # 超参
        # for optimizer(SGD)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        # for lr scheduler(ReduceLROnPlateau)
        self.gamma = gamma
        self.patience = patience
        self.cooldown = cooldown

        # 其他
        self.save_hyperparameters()
        self.accuracy = pl.metrics.Accuracy()

        # 组网
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 1 * 1, 2)

    def forward(self, x):
        # print(x.shape)
        prep = F.conv2d(x, KV, padding=2)
        # print(prep.shape)
        out = F.tanh(self.bn1(torch.abs(self.conv1(prep))))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.tanh(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn3(self.conv3(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn4(self.conv4(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = F.relu(self.bn5(self.conv5(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = F.softmax(out, dim=1)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.gamma, patience=self.patience, cooldown=self.cooldown)
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
