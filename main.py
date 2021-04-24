'''
Author: your name
Date: 2021-04-23 20:52:45
LastEditTime: 2021-04-23 21:44:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /steganography_platform_pl/main.py
'''
# %%
import os
import comet_ml
import numpy as np
import pytorch_lightning as pl
import torch
from icecream import ic
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from src.config import args
from src.datasetmgr import ImageDataModule
from src.models import ZhuNet

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename='zhunet-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)

comet_logger = CometLogger(
    api_key="6vfLO89GXYkYrGcritIRFfqmj",
    save_dir='comet_log',
    project_name="zhunet_project",
    workspace="kevin2li",
    experiment_name='zhunet_v2'  # Optional
)
# %%
datamodule = ImageDataModule()
datamodule.setup()
model = ZhuNet()
trainer = pl.Trainer(gpus='0', max_epochs=250, progress_bar_refresh_rate=1, logger=comet_logger,  callbacks=[checkpoint_callback], auto_lr_find=True)

# %%
# trainer.tune(model, datamodule=datamodule)
# %%
trainer.fit(model, datamodule=datamodule)
# %%
checkpoint_path = 'comet_log/zhunet_project/c2a9d60f268b4c5ab21a837044f986ba/checkpoints'
trainer.logger.experiment.log_model('checkpoint', checkpoint_path)
trainer.logger.experiment.log_asset('archive.zip')
# %%
trainer.test(model, datamodule=datamodule)
