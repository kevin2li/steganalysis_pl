'''
Author: your name
Date: 2021-04-23 20:52:45
LastEditTime: 2021-04-23 21:44:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /steganography_platform_pl/main.py
'''
# %%
import numpy as np
import pytorch_lightning as pl
import torch
from icecream import ic

from src.config import args
from src.datasetmgr import getDataLoader
from src.models import ZhuNet
# %%
train_loader, test_loader = getDataLoader(args)
model = ZhuNet()
trainer = pl.Trainer(model, fast_dev_run=True)

# %%
trainer.fit(model, train_loader, test_loader)

# %%
