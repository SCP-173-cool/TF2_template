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


from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from tqdm import tqdm

## DATASETS SETTING
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):
    """
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

class METRICS(object):
    
    ## Train process Metrics
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    
    ## Train process Metrics
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    METRICS.train_loss(loss)
    METRICS.train_accuracy(labels, predictions)
    
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    
    METRICS.test_loss(t_loss)
    METRICS.test_accuracy(labels, predictions)
    
EPOCHS=5
for epoch in tqdm(range(EPOCHS)):
    for images, labels in train_ds:
        train_step(images, labels)
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
        
    template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss:{}, Test Accuracy: {}"
    print(template.format(epoch+1,
                          METRICS.train_loss.result(),
                          METRICS.train_accuracy.result()*100,
                          METRICS.test_loss.result(),
                          METRICS.test_accuracy.result()*100))
    