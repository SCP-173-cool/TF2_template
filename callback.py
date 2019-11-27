#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:28:17 2019

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True

import tensorflow as tf

def get_callbacks(cfg):

    cb_ModelCheckPoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = cfg.MCP_PATH,
        monitor = cfg.MCP_MONITOR,
        mode = cfg.MCP_MODE,
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    
    cb_TensorBoard = tf.keras.callbacks.TensorBoard(
        log_dir = cfg.TB_PATH,
        write_graph = True,
        update_freq = "batch")
    
    cb_LearningRateDecay = cb_lrschedule(cfg)
    
    
    CALLBACK_LST = [cb_ModelCheckPoint, cb_TensorBoard, cb_LearningRateDecay]
    return CALLBACK_LST

def cb_lrschedule(cfg):
    import numpy as np
    def lr_schedule(epoch):
        PI = 3.1415926
        CYCLE = cfg.LRD_CYCLE
        LR_INIT = cfg.LRD_INIT
        LR_MIN = cfg.LRD_MIN
        lr = ((LR_INIT - LR_MIN) / 2) * (np.cos(PI*(np.mod(epoch-1, CYCLE) / (CYCLE))) + 1) + LR_MIN
        return lr
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)