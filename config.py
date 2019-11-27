#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:24:34 2019

@author: loktarxiao
"""
import sys
sys.dont_write_bytecode = True

import os

class base_config(object):
    EXPERIMENT_NAME = "test-1"
    SAVE_PATH = os.path.join("./logs", EXPERIMENT_NAME)
    
    ## --- common --- ##
    VISIBLE_DEVICES = "0, 1, 2, 3"
    NUM_EPOCHS = 5000
    IMAGES_PER_GPU = 64
    GPU_COUNT = 4
    
    ## --- tf.data.Dataset --- ##
    TRAIN_TFRECORD_LST = ['./tools/train.tfrecord']
    VALID_TFRECORD_LST = ['./tools/valid.tfrecord']
    NUM_TRAIN_SAMPLES  = 80423
    NUM_VALID_SAMPLES  = 26847
    num_processors = 32
    
    ## --- model config --- ##
    input_shape    = (224, 224, 3)
    num_classes    = 22
    
    ## --- callbacks config --- ##
    # ModelCheckPoint
    MCP_MONITOR = "val_Accuracy"
    MCP_MODE = "max"
    MCP_FORMAT = "model-{epoch:02d}-{val_Accuracy:.4f}.h5"
    
    # Tensorboard
    TB_PATH = "TB_events"
    
    # Learning Rate Decay
    LRD_INIT = 0.1
    LRD_CYCLE = 10
    LRD_MIN = 1e-6
    
    
    def __init__(self):
        """Set values of computed attributes."""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.VISIBLE_DEVICES
        
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        self.TRAIN_STEP = self.NUM_TRAIN_SAMPLES / self.BATCH_SIZE
        self.VALID_STEP = self.NUM_VALID_SAMPLES / self.BATCH_SIZE
        
        os.makedirs(os.path.join(self.SAVE_PATH, "weights"), exist_ok=True)
        self.MCP_PATH = os.path.join(self.SAVE_PATH, "weights", self.MCP_FORMAT)
        self.TB_PATH = os.path.join(self.SAVE_PATH, self.TB_PATH)
        
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        print("====================================================")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("====================================================")
        print("\n")