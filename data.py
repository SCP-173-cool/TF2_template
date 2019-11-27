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

def _parse_function(example_proto):
    """Resolve the TFRECORD string message to tensors.
    """
    dics = {
        'image': tf.io.FixedLenFeature([], dtype=tf.string),
        'label1': tf.io.FixedLenFeature([], dtype=tf.int64),
        'label2': tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example_proto, features=dics)

    image = tf.reshape(tf.io.decode_raw(
        parsed_example['image'], tf.uint8), (224, 224, 3))
    label = parsed_example['label1']

    image = tf.cast(image, tf.uint8)
    label = tf.cast(label, tf.float32)

    return image, label

def dataset_from_tfrcord(tfrecord_lst, num_processors=8):
    """Create `tf.dataset` by tfrecord
    Args:
        tfrecord_lst(list): a list included all paths of tfrecord
        num_processors(int): number of processor to load data.
    """

    dataset = tf.data.TFRecordDataset(tfrecord_lst)
    dataset = dataset.map(_parse_function, num_processors)

    return dataset

@tf.function
def train_process_func(image, label):

    image = tf.cast(image, tf.float32) / 255.
    #image = tf.image.random_crop(image, size=[224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.5)
    image = tf.image.random_saturation(image, 0.1, 1.6)
    image = tf.image.random_contrast(image, 0.2, 1.3)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_jpeg_quality(image, 40, 100)
    return image, label

@tf.function
def valid_process_func(image, label):
    image = tf.cast(image, tf.float32) / 255.
    #image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    return image, label

def data_loader(tfrecord_lst,
                shuffle=False,
                batch_size=128,
                num_processors=4,
                process_func=None,
                device='cpu:0'):
    
    """ Create an Iterator to load data
    """

    with tf.device('/{}'.format(device)):

        dataset = dataset_from_tfrcord(tfrecord_lst, num_processors)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        if process_func is not None:
            dataset = dataset.map(process_func, num_processors)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def get_datasets(cfg):
    
    
    train_ds = data_loader(cfg.TRAIN_TFRECORD_LST, 
                           shuffle=True, 
                           batch_size=cfg.BATCH_SIZE,
                           process_func=train_process_func,
                           num_processors=cfg.num_processors)

    valid_ds = data_loader(cfg.VALID_TFRECORD_LST, 
                           shuffle=False, 
                           batch_size=cfg.BATCH_SIZE,
                           process_func=valid_process_func,
                           num_processors=cfg.num_processors)
    
    return train_ds, valid_ds

if __name__ == "__main__":
    """
    """
    from tqdm import tqdm
    train_tfrecord_lst = ['./tools/train.tfrecord']
    train_ds = data_loader(train_tfrecord_lst, 
                           shuffle=True, 
                           batch_size=32,
                           process_func=train_process_func,
                           num_processors=32)

    for images, labels in tqdm(train_ds):
        break

    images = images.numpy()
    print(images.shape)

    import matplotlib.pyplot as plt
    #images = (images + 1) / 2.
    for i in range(len(images)):
        image = images[i]
        plt.imshow(image)
        plt.show()

