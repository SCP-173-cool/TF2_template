#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:24:34 2019

@author: loktarxiao
"""
import sys
sys.dont_write_bytecode = True

import cv2
import numpy as np
import random

def normalize(img):
    img = img.astype(np.float32)
    img = img /127.5
    img = img - 1 
    return img 

def image_read(img_path, mode="rgb"):
    """ The faster image reader with opencv API
    """
    with open(img_path, 'rb') as fp: 
        raw = fp.read()
        if mode == "rgb":
            img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_COLOR)
            img = img[:,:,::-1]
        elif mode == "gray":
            img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_GRAYSCALE)

    return img 

def random_crop(image, crop_shape=(224, 224)):
    oshape = np.shape(image)

    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return image_crop

def resize_shorter_edge(image, size=250):
    """ Resize image to target size 
    """
    h, w = image.shape[:2]
    scale = size / min(h, w)

    new_shape = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
