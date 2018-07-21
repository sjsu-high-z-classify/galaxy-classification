#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main program file

The goal of this project is to create an automated way to catalogue and
classify observed galaxies, with a goal of classifying all unclassified
galaxies for which we have observations. This project acheives this through the
use of a Convolutional Neural Network (CNN) trained using galaxies from the
Galaxy Zoo catalogue.

This file is the main file of the project, from which all the other modules in
the project are called.

Created on Fri Jul  6 11:52:11 2018

@author: J. Andrew Casey-Clyde (jacaseyclyde)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import pandas as pd

import tensorflow as tf

import pipeline
import model
import database

BATCH_SIZE = 100
RECORDS = 10


def main():
    """
    The main function.

    This is the main function. It automatically populates the catalogue if
    necessary, then starts training the model.

    TODO:
        * Add evaluation section
        * Add prediction section
        * Make catalogue check/population more robust, or add a flag
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Add command line flags
    parser.add_argument('-u', '--username', dest='username', help="User name")
    parser.add_argument('-p', '--password', dest='password', help="Password")
    
    args = parser.parse_args()

    # Open the catalogue, or create and open the catalogue if necessary
    try:
        gal_data = pd.read_csv('catalogue.csv')
    except FileNotFoundError:
        database.dataquery(RECORDS, args.username, args.password)
        gal_data = pd.read_csv('catalogue.csv')

    gal_data = pd.DataFrame(gal_data)

#    my_feature_columns = []
#    for key in ['Image']:
#        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.Estimator(
        model_fn=model.cnn_model,
        params={
            # number of classes is the number of unique labels
            'n_classes': len(np.unique(gal_data.Gtype)),
            # 'feature_columns': my_feature_columns
            })

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    classifier.train(
        input_fn=lambda: pipeline.train_input_fn(gal_data, BATCH_SIZE),
        steps=4,
        hooks=[logging_hook])

    return classifier


if __name__ == '__main__':
    main()
