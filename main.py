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
import model

import pandas as pd

import tensorflow as tf

batch_size = 100


def main():
    gal_data = pd.read_csv('Book.csv')
    gal_data = pd.DataFrame(gal_data)

    my_feature_columns = []
    for key in ['Image']:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.Estimator(
            model_fn=model.cnn_model,
            params={
                    'feature_columns': my_feature_columns
                    })

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    classifier.train(
            input_fn=lambda: pipeline.train_input_fn(gal_data, batch_size),
            steps=300,
            hooks=[logging_hook])

    return classifier


if __name__ == '__main__':
    classifier = main()
