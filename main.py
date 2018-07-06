#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:52:11 2018

@author: jacaseyclyde
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pipeline

import pandas as pd
import tensorflow as tf


def main():
    galData = pd.read_csv('Book.csv')
    galData = pd.DataFrame(galData)

    dataset = tf.data.Dataset.from_tensor_slices((galData.RA.tolist(),
                                                  galData.DEC.tolist(),
                                                  galData.TYPE.tolist()))

    dataset = dataset.map(
            lambda ra, dec, galType: tuple(
                    tf.py_func(pipeline.distorted_inputs,
                               [ra, dec, galType],
                               [tf.uint8, tf.int64])))

    return dataset


if __name__ == '__main__':
    dataset = main()

    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()

    sess = tf.Session()

    with sess.as_default():
        f_res = features.eval()
        l_res = labels.eval()

        print(f_res.shape, l_res)
