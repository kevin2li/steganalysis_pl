'''
Author: Kevin Li
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
from src.datasetmgr import ImageDataModule
from src.models import ZhuNet, YedNet

hparams = {
    # path
    'data_dirs': ['data/bb_wow_0.4', 'data/cover'],
    # optimizer(SGD)
    'lr': 0.005,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    # lr scheduler(ReduceLROnPlateau)
    'gamma': 0.2,
    'patience': 20,
    'cooldown': 5,
    # other
    'gpus': '0',
    'seed': 2021,
    'batch_size': 32,
    'max_epochs': 250,
    # comet.ml experiment description
    'api_key': '6vfLO89GXYkYrGcritIRFfqmj',
    'save_dir': 'comet_log',
    'workspace': 'kevin2li',
    'project_name': 'yednet_project',
    'experiment_name': 'yednet_v0',
    'experiment_key': None
}

# %%
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename='{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)

comet_logger = CometLogger(
    api_key=hparams['api_key'],
    save_dir=hparams['save_dir'],
    project_name=hparams['project_name'],
    workspace=hparams['workspace'],
    experiment_name=hparams['experiment_name'],
    experiment_key=hparams['experiment_key'],
)

experiment_id = comet_logger.experiment.id
# %%
datamodule = ImageDataModule(**hparams)
datamodule.setup()
# model = ZhuNet(**hparams)
model = YedNet(**hparams)
trainer = pl.Trainer(gpus=hparams['gpus'], max_epochs=hparams['max_epochs'], progress_bar_refresh_rate=1, logger=comet_logger,  callbacks=[checkpoint_callback], auto_lr_find=True)

# %%
# trainer.tune(model, datamodule=datamodule)
# %%
trainer.fit(model, datamodule=datamodule)
# %%
checkpoint_path = f"{hparams['save_dir']}/{hparams['project_name']}/{experiment_id}/checkpoints"
trainer.logger.experiment.log_model('checkpoint', checkpoint_path)
trainer.logger.experiment.log_asset('code.zip')
# %%
trainer.test(model, datamodule=datamodule)

# %%
model = model.load_from_checkpoint('comet_log/zhunet_project/8d3dd674da27452cb5e4aa17f5379869/checkpoints/zhunet-epoch=210-val_loss=0.44-val_acc=0.85.ckpt')
# %%
trainer.test(model, datamodule=datamodule)
# %%
test_dataloader = datamodule.test_dataloader()
# %%
len(test_dataloader)
# %%
x, y = iter(test_dataloader).next()
x.shape
# %%
y
# %%
model(x)
# %%
import matplotlib.pyplot as plt
plt.imshow(x[7], cmap='gray')
# %%
y
# %%
