#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:28:17 2019

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True

import tensorflow as tf

def get_optimizer():
    """
    """
    optimizer = tf.keras.optimizers.SGD()
    return optimizer


        