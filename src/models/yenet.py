#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__all__ = ['YeNet']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

srm_16filters = np.load('src/utils/srm_16_filters.npy')
srm_minmax = np.load('src/utils/minmax_filters.npy')
srm_filters = np.concatenate((srm_16filters, srm_minmax),axis=0)

srm_filters = torch.from_numpy(srm_filters).to(device=device, dtype=torch.float)
srm_filters = torch.autograd.Variable(srm_filters, requires_grad=False)

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
        self.accuracy = pl.metrics.Accuracy()

        # 组网
        self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)
        self.conv2 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(30, 32, kernel_size=5, stride=1, padding=0)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=3, padding=0)
        self.fc = nn.Linear(16 * 3 * 3, 2)

    def forward(self, x):
        out = self.tlu(F.conv2d(x, srm_filters))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = F.relu(self.conv5(out))
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
        out = F.relu(self.conv6(out))
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
        out = F.relu(self.conv7(out))
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
        out = F.relu(self.conv8(out))
        out = F.relu(self.conv9(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = F.softmax(out, dim=1)
        return out

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
