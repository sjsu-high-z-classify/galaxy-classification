#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is GalNet, a Convolutional Neural Network (CNN) made to classify galaxies.
Copyright (C) 2018  J. Andrew Casey-Clyde, Jean Donet, and Hiren Thummar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import shutil
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

import pipeline
import model
import database


def download_db(records, username, password):
    """Download the catalogue.

    This function starts a user prompt, asking for a CasJobs login, as well as
    how many records to download. Catalogue will be saved in catalogue.csv.

    Returns:
        DataFrame: A pandas DataFrame object containing the catalogue.
    """
    database.dataquery(records, username, password)

    return pd.read_csv('./catalogue.csv')


def main(argv):
    """The main function.

    This is the main function. Note that default values for all flags are set
    outside of this definition, therefore values for all attributes of argv
    will need to be initialized if not called directly (e.g. through
    `$ python galnet.py -T`).

    Args:
        argv: An argparse Namespace object with the following attributes.
            BATCH_SIZE (int): Batch size to use with the dataset.
            TEST_SIZE (float): Proportion of dataset to use for testing.
            TRAIN (bool): Train and test the CNN.
            EPOCHS (int): Number of epochs to train the network for.
            PRED (bool): Make classification predictions on a dataset
            RESET (bool): Resets the entire neural network, including catalogue
            DB (bool): Resets (deletes) the catalogue

    Todo:
        * Add prediction section
    """

    # Check flags
    if argv.RESET:
        # resets the entire neural network and catalogue
        shutil.rmtree('./model/checkpoints')
        shutil.rmtree('./catalogue.csv')
    elif argv.DB:
        shutil.rmtree('./catalogue.csv')

    # Exit the program if either reset happened, unless flagged otherwise
    if (argv.RESET or argv.DB) and not (argv.TRAIN or argv.PRED):
        sys.exit()

    # Open the catalogue, or create and open the catalogue if necessary
    try:
        gal_data = pd.read_csv('catalogue.csv')
    except FileNotFoundError:
        print("Couldn't find galaxy catalogue. You will need to populate the "
              "catalogue before proceeding further.")
        gal_data = download_db(argv.RECORDS, argv.USERNAME, argv.PASSWORD)

    # Ensures this is a DataFrame, since read_csv can also return a TextStream
    gal_data = pd.DataFrame(gal_data)

    # populate label dictionary
    g_types = np.unique(gal_data.Gtype.astype(str).tolist())
    g_dict = dict(enumerate(g_types))

    g_dict = dict((v, k) for k, v in g_dict.items())

    gal_data['Gtype'] = gal_data['Gtype'].map(g_dict)

    # Instantiate the cnn
    classifier = tf.estimator.Estimator(
        model_fn=model.cnn_model,
        model_dir='./model/checkpoints',
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
                                             test_size=argv.TEST_SIZE)

    if argv.TRAIN:
        # Training
        classifier.train(
            input_fn=lambda: pipeline.train_input_fn(train_data,
                                                     argv.BATCH_SIZE),
            steps=argv.EPOCHS,
            hooks=[logging_hook])

        # Evaluation
        eval_results = classifier.evaluate(
            input_fn=lambda: pipeline.eval_input_fn(
                test_data, argv.BATCH_SIZE))

        print(eval_results)


if __name__ == '__main__':
    # Parse command line arguments
    PARSER = argparse.ArgumentParser()

    # Add command line flags
    PARSER.add_argument('-u', '--User', action='store',
                        type=str, dest='USERNAME', default="panda1")
    PARSER.add_argument('-p', '--Password', dest='PASSWORD', action='store',
                        type=str, default="Panda")
    PARSER.add_argument('-r', '--records', dest='RECORDS', action='store',
                        help='number of objects to download from SDSS',
                        default=1000)
    PARSER.add_argument('-b', '--batch_size', dest='BATCH_SIZE',
                        help='Batch Size.', default=100)
    PARSER.add_argument('-t', '--test_size', dest='TEST_SIZE',
                        help='Proportion of dataset to use for testing '
                        '(default 0.25).',
                        default=0.25)
    PARSER.add_argument('-T', '--train', dest='TRAIN', action='store_true',
                        help='Train and test the neural network.')
    PARSER.add_argument('-E', '--epochs', dest='EPOCHS', default=100,
                        help='Number of epochs to train the network for '
                        '(default 100).')
    PARSER.add_argument('-P', '--predict', dest='PRED', action='store_true',
                        help='Use the CNN to make predictions.')
    PARSER.add_argument('-R', '--reset', dest='RESET', action='store_true',
                        help='Reset the entire neural network.')
    PARSER.add_argument('-D', '--db_reset', dest='DB', action='store_true',
                        help='Reset the database.')

    ARGS = PARSER.parse_args()
    main(ARGS)
