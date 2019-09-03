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

def image_read(img_path):
    """ The faster image reader with opencv API
    """
    with open(img_path, 'rb') as fp:
        raw = fp.read()
        img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_COLOR)
        img = img[:,:,::-1]

    return img

def resize_shorter_edge(image, size=250):
    """ Resize image to target size 
    """
    h, w = image.shape[:2]
    scale = size / min(h, w)

    new_shape = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
