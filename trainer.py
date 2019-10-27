#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:24:34 2019

@author: loktarxiao
"""
import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from tqdm import tqdm

class Trainer(object):
    """
    """
    def __init__(self, datasets, model, config, metrics, optimizers):
        self.train_ds, self.valid_ds = datasets
        self.model = model
        self.cfgs = config
        self.mets = metrics
        self.opts = optimizers

        self.train_variables = self.model.trainable_variables

    def _compile(self):
        """
        """
        self.model.compile(optimizer=self.opts.OPTIMIZER,
                           loss=self.opts.LOSS_FUNC,
                           metrics=self.mets.METRICS_LST)
    
    def train(self):
        """
        """
        self.model.fit(self.train_ds, 
                       epochs=self.cfgs.EPOCHS,
                       validation_data=self.valid_ds)
    
    """
    #@tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.opts.LOSS_FUNC(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opts.OPTIMIZER.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.mets.train_loss(loss)
        self.mets.train_accuracy(labels, predictions)


    #@tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        loss = self.opts.LOSS_FUNC(labels, predictions)
    
        self.mets.test_loss(loss)
        self.mets.test_accuracy(labels, predictions)


    def run_loop(self):
        EPOCHS = self.cfgs.EPOCHS
        for epoch in tqdm(range(EPOCHS)):
            for images, labels in self.train_ds:
                template = "Epoch {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}"
                self.train_step(images, labels)
                print(template.format(epoch+1,
                                      self.mets.train_loss.result(),
                                      self.mets.train_accuracy.result()*100,))
            for images, labels in self.valid_ds:
                self.test_step(images, labels)

            template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss:{}, Test Accuracy: {}"
            print(template.format(epoch+1,
                                  self.mets.train_loss.result(),
                                  self.mets.train_accuracy.result()*100,
                                  self.mets.test_loss.result(),
                                  self.mets.test_accuracy.result()*100))

            self.mets.reset_all_metrics()

    """