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

import sys
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

import pipeline
import model
import database


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
    parser.add_argument('-u', '--username', dest='USERNAME', help='User name')
    parser.add_argument('-p', '--password', dest='PASSWORD', help='Password')
    parser.add_argument('-B', '--batch_size', dest='BATCH_SIZE',
                        help='Batch Size', default=100)
    parser.add_argument('-R', '--records', dest='RECORDS', help='Number of'
                        ' records, or total dataset size')
    parser.add_argument('-t', '--test_size', dest='TEST_SIZE',
                        help='Proportion of dataset to use for testing',
                        default=0.25)

    args = parser.parse_args()

    # Populate the catalogue
    if args.RECORDS is not None:
        try:
            database.dataquery(args.RECORDS, args.USERNAME, args.PASSWORD)
        except Exception as error:
            print(error)
            sys.exit("Please check username and/or password")

    # Open the catalogue, or create and open the catalogue if necessary
    try:
        gal_data = pd.read_csv('catalogue.csv')
    except FileNotFoundError:
        sys.exit('No catalogue initialized. Please rerun the program and '
                 'specify the size of the dataset to use, as well as a CasJobs'
                 'username and password.')

    gal_data = pd.DataFrame(gal_data)

    # populate label dictionary
    g_types = np.unique(gal_data.Gtype.astype(str).tolist())
    g_dict = dict(enumerate(g_types))

    g_dict = dict((v, k) for k, v in g_dict.items())

    gal_data['Gtype'] = gal_data['Gtype'].map(g_dict)

    # Instantiate the cnn
    classifier = tf.estimator.Estimator(
        model_fn=model.cnn_model,
        params={
            # number of classes is the number of unique labels
            'n_classes': len(np.unique(gal_data.Gtype)),
            })

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Split data into train and test sets
    train_data, test_data = train_test_split(gal_data,
                                             test_size=args.TEST_SIZE)

    # Training
    classifier.train(
        input_fn=lambda: pipeline.train_input_fn(train_data, args.BATCH_SIZE),
        steps=4,
        hooks=[logging_hook])

    # Evaluation
    eval_results = classifier.evaluate(
        input_fn=lambda: pipeline.eval_input_fn(
            test_data, args.BATCH_SIZE))

    print(eval_results)

    return eval_results


if __name__ == '__main__':
    main()
