#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:24:34 2019

@author: loktarxiao
"""
import sys
sys.dont_write_bytecode = True

from config import base_config

from optimizer import get_optimizer
from loss import get_loss_function
from model import get_model
from metrics import get_metrics_lst
from callback import get_callbacks
from trainer import Trainer
from data import get_datasets

if __name__ == "__main__":
    
    config = base_config()
    config.METRICS_LST = get_metrics_lst()
    config.OPTIMIZER = get_optimizer()
    config.LOSS_FUNC = get_loss_function()
    config.CALLBACK_LST = get_callbacks(config)
    
    config.display()

    model       = get_model(config)
    datasets    = get_datasets(config)
    trainer     = Trainer(datasets, model, config)

    trainer._compile()
    trainer.train()
