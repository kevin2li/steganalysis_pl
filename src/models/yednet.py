
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.utils import ABS, HPF, TLU, SPPLayer
from torchmetrics import Accuracy

__all__ = ['YedNet']

class YedNet(pl.LightningModule):
    def __init__(self, lr: float=0.005, weight_decay: float= 5e-4, gamma: float = 0.2, momentum: float = 0.9, step_size: int = 50, **kwargs):
        super(YedNet, self).__init__()
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
            nn.Linear(1024, 2),
            # nn.Softmax(dim=1)
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
        params_wd, params_rest = [], []
        for m in self.parameters():
            if m.requires_grad:
                (params_wd if m.dim()!=1 else params_rest).append(m)
        param_groups = [{'params': params_wd, 'weight_decay': self.weight_decay},
                        {'params': params_rest}]
        optimizer = torch.optim.SGD(param_groups, lr=self.lr, momentum=self.momentum)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.gamma, patience=self.patience, cooldown=self.cooldown)
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
        self.log('test_acc', test_acc, prog_bar=True, on_step=False, on_epoch=True)
        return {'test_loss': test_loss, 'test_acc': test_acc}
