#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:28:17 2019

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True

import tensorflow as tf

class Metrics(object):
    
    ## Train process Metrics
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    
    ## Train process Metrics
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    all_metrics = [train_loss, train_accuracy, test_loss, test_accuracy]

    def reset_all_metrics(self):
        for i in self.all_metrics:
            i.reset_states()