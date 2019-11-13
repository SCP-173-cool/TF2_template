#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:24:34 2019

@author: loktarxiao
"""
import sys
sys.dont_write_bytecode = True

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from img_utils import image_read, resize_shorter_edge


def _int64_feature(value):
    """Conver integer data to a string which is accepted by tensorflow record.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Conver byte data to a string which is accepted by tensorflow record.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _tfrecord_string(feature):
    """Convert image array and label message to tensorflow serialized string.
    Args:
        feature: tensorflow features from messages
    Returns:
        tf_serialized: tensorflow serialized string message matched by `TFRECORD`.
    """
    tf_features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=tf_features)
    tf_serialized = example.SerializeToString()
    return tf_serialized



def MessageProcess(item):
    image_path, label = item[0], item[1]
    image = image_read(image_path)
    image = resize_shorter_edge(image, size=250)

    feature = {
        'label': _int64_feature([label]),
        'image': _bytes_feature([image.tobytes()]),
        'image_shape': _int64_feature(list(image.shape))
    }
    return feature

def RecorderMessage(item):
    """Read image and convert it to serialized string.
    Args:
        item(list): 
    Returns:
        tf_serialized_string: The item tfrecord string.
    """
    feature = MessageProcess(item)
    tf_serialized_string = _tfrecord_string(feature)
    return tf_serialized_string

def RecordMaker(item_lst, writer_path, num_processes=5):
    """
    """
    writer = tf.io.TFRecordWriter(writer_path)
    pool = Pool(processes=num_processes)
    results = []
    for item in item_lst[:]:
        results.append(pool.apply_async(RecorderMessage, args=(item,)))
    pool.close()
    pool.join()

    print("[INFO] Tasks has been distributed to each pool.")
    print("[WAITING] ......")
    for result in tqdm(results):
        writer.write(result.get())
    print("[INFO] Tasks has been completed.")
    writer.close()

if __name__ == "__main__":
    import os
    ROOT_DIR = "/home/loktar/Datasets/dogs_vs_cats/"
    train_lst = []
    valid_lst = []
    with open(os.path.join("./", "message_lst.csv")) as fp:
        messages = fp.readlines()
    print(messages[:3])
    for i in messages:
        rela_path, label, image_type = i.strip().split(",")
        item = [os.path.join(ROOT_DIR, rela_path) , int(label)]
        if image_type == "train":
            train_lst.append(item)
        elif image_type == "valid":
            valid_lst.append(item)
    
    RecordMaker(train_lst, "./train.tfrecord", num_processes=8)
    RecordMaker(valid_lst, "./valid.tfrecord", num_processes=8)
