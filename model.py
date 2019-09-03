#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:24:34 2019

@author: loktarxiao
"""
import sys
sys.dont_write_bytecode = True

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras import Model

import tensorflow as tf


class MyModel(Model):
    """
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.pool1 = MaxPooling2D((5, 5))
        self.flatten = Flatten()
        self.dp1 = Dropout(0.5)
        self.d1 = Dense(128, activation="relu")
        self.dp2 = Dropout(0.2)
        self.d2 = Dense(2, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dp1(x)
        x = self.d1(x)
        x = self.dp2(x)
        return self.d2(x)

def residual_block(input_tensor, block_type, n_filters):
    shortcut = input_tensor
    if block_type == 'conv':
        strides = 2
        shortcut = tf.keras.layers.Conv2D(filters=n_filters,
                                          kernel_size=1,
                                          padding='same',
                                          strides=strides,
                                          kernel_initializer='he_normal',
                                          use_bias=False,
                                          kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(momentum=0.9)(shortcut)

    elif block_type == 'identity':
        strides = 1

    x = tf.keras.layers.Conv2D(filters=n_filters,
                               kernel_size=3,
                               padding='same',
                               strides=strides,
                               kernel_initializer='he_normal',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=n_filters,
                               kernel_size=3,
                               padding='same',
                               strides=1,
                               kernel_initializer='he_normal',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x


def ResNet34(input_shape=[None, None, 3], num_classes=1000, include_top=True, return_endpoints=False):
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               padding='same',
                               strides=2,
                               kernel_initializer='he_normal',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(3, 2, padding='same')(x)

    for n_filters, reps, downscale in zip([64, 128, 256, 512],
                                          [3, 4, 6, 3],
                                          [False, True, True, True]):
        for i in range(reps):
            if i == 0 and downscale:
                x = residual_block(input_tensor=x,
                                   block_type='conv',
                                   n_filters=n_filters)
            else:
                x = residual_block(input_tensor=x,
                                   block_type='identity',
                                   n_filters=n_filters)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if include_top:
        x = tf.keras.layers.Dense(units=num_classes)(x)
    return tf.keras.Model(inputs=input_tensor, outputs=x, name='ResNet34')