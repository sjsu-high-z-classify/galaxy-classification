#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:53:52 2018

@author: jacaseyclyde
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

import imageio

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


def get_image(galObj):
    """
    Reads and parses examples from image data files. Expects data in the form
    of rgb images of any extension.
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.
    Args:
        filename_queue: A queue of strings with the filenames to read from.
    Returns:
        image: a [height, width, depth] uint8 Tensor with the image data
        label: an int32 Tensor with the label in the range 0..9.
    """

    # Get image data for for object with ID objID
    image = imageio.imread('http://skyserver.sdss.org/dr14/SkyServerWS/'
                           'ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image'
                           '&ra={0}'
                           '&dec={1}'
                           '&scale=.2'
                           '&width=200'
                           '&height=200'.format(galObj.RA, galObj.DEC))

    return image, label


def distorted_inputs(galObj):
    """Construct distorted input for CIFAR training using the Reader ops.
    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
        size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    image, label = get_image(galObj)

    # From here is where we can start to apply distortions. We should first
    # do some research on what distortions may be appropriate (though flipping
    # seems reasonable)

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
#
#    # Because these operations are not commutative, consider randomizing
#    # the order their operation.
#    # NOTE: since per_image_standardization zeros the mean and makes
#    # the stddev unit, this likely has no effect see tensorflow#1458.
#    distorted_image = tf.image.random_brightness(distorted_image,
#                                                 max_delta=63)
#    distorted_image = tf.image.random_contrast(distorted_image,
#                                               lower=0.2, upper=1.8)

    # Generate a batch of images and labels by building up a queue of examples.
    return image, label

