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
from src.models import ZhuNet, YedNet, YeNet, SRNet, XuNet

hparams = {
    # path
    'data_dirs': ['data/bb_suniward_0.4', 'data/cover'],
    # optimizer(SGD)
    'lr': 0.005,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    # lr scheduler(ReduceLROnPlateau)
    'gamma': 0.2,
    'patience': 25,
    'cooldown': 5,
    # other
    'gpus': '0',
    'seed': 2021,
    'batch_size': 32,
    'max_epochs': 320,
    # comet.ml experiment description
    'api_key': '6vfLO89GXYkYrGcritIRFfqmj',
    'save_dir': 'comet_log',
    'workspace': 'kevin2li',
    'project_name': 'zhunet_project',
    'experiment_name': 'zhunet_suniward',
    'experiment_key': None
}
pl.seed_everything(hparams['seed'])
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
comet_logger.experiment.log_parameters(hparams)
# %%
datamodule = ImageDataModule(**hparams)
datamodule.setup()

model = ZhuNet(**hparams)
# model = YedNet(**hparams)
# model = XuNet(**hparams)
# model = YeNet(**hparams)
# model = SRNet(**hparams)

trainer = pl.Trainer(gpus=hparams['gpus'], max_epochs=hparams['max_epochs'], progress_bar_refresh_rate=1, logger=comet_logger,  callbacks=[checkpoint_callback], auto_lr_find=True)

# %%
# trainer.tune(model, datamodule=datamodule)
# %%
trainer.fit(model, datamodule=datamodule)

# %%
# resume training
# checkpoint_path = f"{hparams['save_dir']}/{hparams['project_name']}/{experiment_id}/checkpoints/last.ckpt"
# trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, gpus=hparams['gpus'], max_epochs=hparams['max_epochs']+80, progress_bar_refresh_rate=1, logger=comet_logger,  callbacks=[checkpoint_callback])
# trainer.fit(model, datamodule=datamodule)
# %%
checkpoint_dir = f"{hparams['save_dir']}/{hparams['project_name']}/{experiment_id}/checkpoints"
trainer.logger.experiment.log_model('checkpoint', checkpoint_dir)
trainer.logger.experiment.log_asset('code.zip')
# %%
trainer.test(model, datamodule=datamodule)

# %%
# model = model.load_from_checkpoint('comet_log/yednet_project/d99bc82909864fe0bd038918147b9a0c/checkpoints/epoch=247-val_loss=0.47-val_acc=0.83.ckpt')
model = model.load_from_checkpoint('comet_log/yednet_project/2c690662c549495d87bbfdba1ef7dfab/checkpoints/epoch=247-val_loss=0.48-val_acc=0.81.ckpt')
trainer.test(model, datamodule=datamodule)

# %%
