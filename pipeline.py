#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The data pipeline, where input fn's are defined and image data retreived
Copyright (C) 2018  J. Andrew Casey-Clyde and Hiren Thummar

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

import imageio

import numpy as np

import tensorflow as tf


def _get_image(ra, dec):
    """Fetches (200x200) px galaxy images from SDSS based on ra and dec."""

    # Get image data for for object with ID objID
    image = imageio.imread('http://skyserver.sdss.org/dr14/SkyServerWS/'
                           'ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image'
                           '&ra={0}'
                           '&dec={1}'
                           '&scale=.2'
                           '&width=200'
                           '&height=200'.format(ra, dec))

    image = image.astype(np.float32)

    return image


def _get_image_record(ra, dec, gal_type):
    """Wrapper function for including the label for each image."""
    image = _get_image(ra, dec)
    return image, gal_type


def _dict_wrapper(image, label):
    """Wraps feature column tensors and labels in dict."""

    return {'Image': image, 'label': label}


def train_input_fn(records, batch_size):
    """Input function for CNN training.

    This function defines how data is handled, processed, and parsed by the CNN
    during the training phase

    Args:
        record: A DataFrame slice representing one record from catalogue.csv
        batch_size: training batch size

    Returns:
        A nested structure of `tf.Tensor` objects for use as the input layer of
        a CNN.
    """

    # Standardizing data types
    ra = records.ra.astype(np.float32).tolist()
    dec = records.dec.astype(np.float32).tolist()
    g_type = records.Gtype.tolist()

    dataset = tf.data.Dataset.from_tensor_slices((ra, dec, g_type))

    dataset = dataset.map(
        lambda ra, dec, g_type: tuple(
            tf.py_func(_get_image_record,
                       [ra, dec, g_type],
                       [tf.float32, tf.int32])))

    # This wrapping has to happen seperately because it needs to operate on the
    # tensors returned by py_func, which wraps the returns of _get_image_record
    dataset = dataset.map(_dict_wrapper)

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(record, batch_size):
    """Evaluation function for CNN testing.

    This function defines how data is handled, processed, and parsed by the CNN
    during the testing phase

    Args:
        record: A DataFrame slice representing one record from catalogue.csv
        batch_size: training batch size

    Returns:
        A nested structure of `tf.Tensor` objects for use as the input layer of
        a CNN.
    """

    # Standardizing data types
    ra = record.ra.astype(np.float32).tolist()
    dec = record.dec.astype(np.float32).tolist()
    g_type = record.Gtype.tolist()

    dataset = tf.data.Dataset.from_tensor_slices((ra, dec, g_type))

    dataset = dataset.map(
        lambda ra, dec, g_type: tuple(
            tf.py_func(_get_image_record,
                       [ra, dec, g_type],
                       [tf.float32, tf.int32])))

    dataset = dataset.map(_dict_wrapper)

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()
