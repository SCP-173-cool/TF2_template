#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:24:34 2019

@author: loktarxiao
"""
import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np

from optimizer import Optimizers
from model import MyModel, ResNet34
from config import MNIST_config
from metric import Metrics
from trainer import Trainer
from data import All_datasets

if __name__ == "__main__":
    
    model       = ResNet34(input_shape=[224, 224, 3], num_classes=2, include_top=True)
    datasets    = All_datasets
    config      = MNIST_config()
    metrics     = Metrics()
    optimizers  = Optimizers()
    trainer     = Trainer(datasets, model, config, metrics, optimizers)

    trainer._compile()
    trainer.train()