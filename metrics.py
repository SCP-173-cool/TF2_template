#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:28:17 2019

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True

import tensorflow as tf

def get_metrics_lst():
    """
    """
    accuracy = Accuracy(mode="sparse")
    metrics_lst = [accuracy]
    return metrics_lst

class Accuracy(tf.keras.metrics.Metric):
    """
    """
    def __init__(self, name='Accuracy', mode="sparse", **kwargs):
        """
        """
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.mode = mode
        self.TP_TN = self.add_weight(name='TP_TN', initializer='zeros')
        self.ALL = self.add_weight(name='ALL', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if tf.rank(y_true) == tf.rank(y_pred):
            y_true = tf.squeeze(y_true, -1)
            
        if self.mode == "sparse":
            y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.float32)

        if self.mode == "categorical":
            y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.float32)
            y_true = tf.cast(tf.math.argmax(y_true, axis=-1), tf.float32)
        
        values = tf.cast(tf.math.equal(y_true, y_pred), tf.float32)
        length = tf.reduce_sum(tf.ones_like(values))
        self.TP_TN.assign_add(tf.math.reduce_sum(values))
        self.ALL.assign_add(length)
    
    def result(self):
        return self.TP_TN / self.ALL

    

