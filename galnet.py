#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is GalNet, a Convolutional Neural Network (CNN) made to classify galaxies.
Copyright (C) 2018  J. Andrew Casey-Clyde, Hiren Thummar, and Jean Donet

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
import getpass

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

import pipeline
import model
import database


def download_db():
    """Download the catalogue.

    This function starts a user prompt, asking for a CasJobs login, as well as
    how many records to download. Catalogue will be saved in catalogue.csv.

    Returns:
        DataFrame: A pandas DataFrame object containing the catalogue.
    """
    print('Populating the database requires a CasJobs login.')
    username = input('Username: ')
    password = getpass.getpass()

    records = int(input('How many records would you like to populate the '
                        'catalogue with? (Default 1e4)') or 1e4)

    try:
        database.dataquery(records, username, password)
    except Exception as error:
        print(error)

        print("Please re-enter username and password.")
        username = input("Username: ")
        password = getpass.getpass()

        try:
            database.dataquery(records, username, password)
        except Exception as error:
            print(error)
            sys.exit("Please check your username and password and re-run this "
                     "application.")

    return pd.read_csv('./catalogue.csv')


def main():
    """The main function.

    This is the main function. It can be run entirely from the command line,
    but if there's no galaxy catalogue, it will prompt for a CasJobs login, as
    well as the number of records to retreive.

    Keyword Args:
        -b, --batch_size: Batch size to use with the dataset
        -t, --test_size: Proportion of dataset to use for testing
        -T, --train: Train and test the CNN
        -E, --epochs: Number of epochs to train the network for
        -P, --predict: Make classification predictions on a dataset
        -R, --reset: resets the entire neural network, including catalogue
        -D, --db_reset: resets (deletes) the catalogue

    Todo:
        * Add prediction section
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Add command line flags
    parser.add_argument('-b', '--batch_size', dest='BATCH_SIZE',
                        help='Batch Size.', default=100)
    parser.add_argument('-t', '--test_size', dest='TEST_SIZE',
                        help='Proportion of dataset to use for testing '
                        '(default 0.25).',
                        default=0.25)
    parser.add_argument('-T', '--train', dest='TRAIN', action='store_true',
                        help='Train and test the neural network.')
    parser.add_argument('-E', '--epochs', dest='EPOCHS', default=100,
                        help='Number of epochs to train the network for '
                        '(default 100).')
    parser.add_argument('-P', '--predict', dest='PRED', action='store_true',
                        help='Use the CNN to make predictions.')
    parser.add_argument('-R', '--reset', dest='RESET', action='store_true',
                        help='Reset the entire neural network.')
    parser.add_argument('-D', '--db_reset', dest='DB', action='store_true',
                        help='Reset the database.')

    args = parser.parse_args()

    # Check flags
    if args.RESET:
        # resets the entire neural network and catalogue
        shutil.rmtree('./catalogue.csv')
    elif args.DB:
        shutil.rmtree('./catalogue.csv')

    # Exit the program if either reset happened, unless flagged otherwise
    if (args.RESET or args.DB) and not (args.TRAIN or args.PRED):
        sys.exit()

    # Open the catalogue, or create and open the catalogue if necessary
    try:
        gal_data = pd.read_csv('./catalogue.csv')
    except FileNotFoundError:
        print("Couldn't find galaxy catalogue. You will need to populate the "
              "catalogue before proceeding further.")
        gal_data = download_db()

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
                                             test_size=args.TEST_SIZE)

    if args.TRAIN:
        # Training
        classifier.train(
            input_fn=lambda: pipeline.train_input_fn(train_data,
                                                     args.BATCH_SIZE),
            steps=4,
            hooks=[logging_hook])

        # Evaluation
        eval_results = classifier.evaluate(
            input_fn=lambda: pipeline.eval_input_fn(
                test_data, args.BATCH_SIZE))

        # save the model

    print(eval_results)

    return eval_results


if __name__ == '__main__':
    main()
