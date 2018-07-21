#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input pipeline

This module contains all input function definitions and the data handline
pipeline

TODO:
    * Redefine get_images to only grab images from online
        * Add label processing to appropriate mapping fn's
    * add input function for evaluation
    * add input function for prediction
    * automatically build class library

Created on Fri Jun 15 14:53:52 2018

@author: jacaseyclyde
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio

import numpy as np

import tensorflow as tf

# =============================================================================
# constants
# =============================================================================

# currently all of these constants are based on the CIFAR-10 example from tf
# all these values will need to change for our actual dataset

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
# TODO: find numbers that describe and are useful for our galaxy dataset
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# {x: x**2 for x in (2, 4, 6)} # potential auto population of type dict later
gal_dict = {'S': 0, 'E': 1, 'UN': 2}


def get_image(ra, dec, gal_type):
    """Fetches galaxy images from SDSS.

    Fetches a 200x200 JPEG image of a galaxy from SDSS servers, converting it
    to an array of RGB values. Additionally parses classification labels for
    training images.

    Args:
        ra: galaxy right ascension
        dec: galaxy declination
        gal_type: type of galaxy, as determined by galaxy zoo

    Returns:
        image: a [height, width, depth] uint8 Tensor with the image data
        label: an int32 Tensor with the label in the range 0..9.

    TODO:
        * Separate this out into it's own function that only handles the image
    """

    # Get image data for for object with ID objID
    image = imageio.imread('http://skyserver.sdss.org/dr14/SkyServerWS/'
                           'ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image'
                           '&ra={0}'
                           '&dec={1}'
                           '&scale=.2'
                           '&width=200'
                           '&height=200'.format(ra, dec))

    image = image.astype(np.float32)

    label = gal_dict[gal_type.decode()]

    return image, label


def distorted_inputs(image, label):
    """
    ***SECTION IN PROGRESS***

    Applies random distortions to input images (flipping vertically and/or
    horizontally) before packaging.

    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
        size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # return images in a dict. this can be useful if we need to pass other data
    # as well
    return {'Image': image}, label


def train_input_fn(record, batch_size):
    """Input function for CNN training.

    This function defines how data is handled, processed, and parsed by the CNN
    during training phase

    Args:
        record: A DataFrame slice representing one record from catalogue.csv
        batch_size: training batch size
    """
    # Standardizing data types
    ra = record.ra.astype(np.float32).tolist()
    dec = record.dec.astype(np.float32).tolist()
    g_type = record.Gtype.tolist()

    dataset = tf.data.Dataset.from_tensor_slices((ra, dec, g_type))

    dataset = dataset.map(
            lambda ra, dec, galType: tuple(
                    tf.py_func(get_image,
                               [ra, dec, galType],
                               [tf.float32, tf.int64])))

    dataset = dataset.map(distorted_inputs)

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

