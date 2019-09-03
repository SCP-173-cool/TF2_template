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
        'image_shape': tf.io.FixedLenFeature(shape=(3, ), dtype=tf.int64),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, features=dics)

    image = tf.reshape(tf.io.decode_raw(
        parsed_example['image'], tf.uint8), parsed_example['image_shape'])
    label = parsed_example['label']

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

    image = tf.image.random_crop(image, size=[224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.5)
    image = tf.image.random_saturation(image, 0.1, 1.6)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 255.
    return image, label

@tf.function
def valid_process_func(image, label):

    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 255.
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

train_tfrecord_lst = ['./tools/train.tfrecord']
train_ds = data_loader(train_tfrecord_lst, 
                        shuffle=True, 
                        batch_size=32,
                        process_func=train_process_func,
                        num_processors=8)

valid_tfrecord_lst = ['./tools/valid.tfrecord']
valid_ds = data_loader(valid_tfrecord_lst, 
                        shuffle=True, 
                        batch_size=32,
                        process_func=valid_process_func,
                        num_processors=8)

All_datasets = (train_ds, valid_ds)

if __name__ == "__main__":
    """
    """
    train_tfrecord_lst = ['./tools/train.tfrecord']
    train_ds = data_loader(train_tfrecord_lst, 
                           shuffle=True, 
                           batch_size=12,
                           process_func=train_process_func,
                           num_processors=8)

    for images, labels in train_ds:
        pass

    images = images.numpy()
    print(images.shape)

    import matplotlib.pyplot as plt
    for i in range(len(images)):
        image = images[i]
        plt.imshow(image)
        plt.show()


"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

All_datasets = (train_ds, test_ds)
"""