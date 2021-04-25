import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from src.utils import ABS, TLU, HPF, SPPLayer

__all__ = ['YedNet']

class YedNet(pl.LightningModule):
    def __init__(self, lr: float=0.005, weight_decay: float= 5e-4, gamma: float = 0.2, momentum: float = 0.9, patience: int = 20, cooldown: int = 5, **kwargs):
        # 超参
        # for optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        # for lr scheduler(ReduceLROnPlateau)
        self.gamma = gamma
        self.patience = patience
        self.cooldown = cooldown

        self.save_hyperparameters()
        self.accuracy = pl.metrics.Accuracy()
        # 组网
        super(YedNet, self).__init__()
        self.hpf = HPF()
        self.group1 = nn.Sequential(
            nn.Conv2d(30, 30, 5, 1, 2),
            ABS(),
            nn.BatchNorm2d(30),
            TLU(3.0)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(30, 30, 5, 1, 2),
            nn.BatchNorm2d(30),
            TLU(3.0),
            nn.AdaptiveAvgPool2d((128, 128))
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(30, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64))
        )
        self.group4 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        self.group5 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classfier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = x.view(-1, 128)
        out = self.classfier(x)
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