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
from glob import glob
from pathlib import Path

import comet_ml
import numpy as np
import pytorch_lightning as pl
import torch
from icecream import ic
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from src.datasetmgr import ImageDataModule
from src.models import SRNet, XuNet, YedNet, YeNet, ZhuNet
from src.utils import initialization

all_stego_dirs = [
    '/home/likai/DataSets/WOWstego(0.4)',
    '/home/likai/DataSets/SUI_stego(0.4)',
    '/home/likai/DataSets/MiPODstego(0.4)',
    '/home/likai/DataSets/HUGOstego(0.4)',
    '/home/likai/DataSets/HILLstego(0.4)',
    '/home/likai/DataSets/MGstego(0.4)',
    '/home/likai/DataSets/MVGstego(0.4)',
    '/home/likai/DataSets/UTGANstego(0.4)',
]

for dir in all_stego_dirs:
    data_dirs = []
    data_dirs.append(dir)
    data_dirs.append('/home/likai/DataSets/COVER')

    hparams = {
        # path
        'data_dirs': data_dirs,
        # optimizer(SGD)
        'lr': 0.005,    # yenet:0.001 others:0.005
        'weight_decay': 5e-4,
        'momentum': 0.9,
        # lr scheduler(ReduceLROnPlateau)
        'gamma': 0.2,
        'patience': 25,
        'cooldown': 5,
        'step_size': 50,
        # other
        'gpus': '0',
        'seed': 2021,
        'batch_size': 32,
        'max_epochs': 280,
        'gradient_clip_val': 1.0,
        # comet.ml experiment description
        'api_key': '6vfLO89GXYkYrGcritIRFfqmj',
        'save_dir': 'comet_log',
        'workspace': 'kevin2li',
        'project_name': 'zhunet_project',
        'experiment_name': f'zhunet_{dir[21:]}',
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

    initialization(model)
    trainer = pl.Trainer(gpus=hparams['gpus'], max_epochs=hparams['max_epochs'], gradient_clip_val=hparams['gradient_clip_val'], progress_bar_refresh_rate=1, logger=comet_logger,  callbacks=[checkpoint_callback], auto_lr_find=True)

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
    checkpoint_list = glob(str(Path(checkpoint_dir) / '*'))
    for checkpoint in checkpoint_list:
        model = model.load_from_checkpoint(checkpoint)
        trainer.test(model, datamodule=datamodule)
    # %%
