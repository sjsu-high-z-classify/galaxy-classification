#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN model definition module.
Copyright (C) 2018  Jean Donet and J. Andrew Casey-Clyde

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

# Importing libraries
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.DEBUG)


def cnn_model(features, mode, params):
    """CNN model function

    This is the singular function of this file, and is where the convolutional
    neural network itself is defined.

    Args:
        features (dict(tf.Tensor)): dict of input features (and labels if
            training or testing)
        mode (tf.estimator.ModeKeys): What mode the model is running in.
        params (dict): A dict of other params, including the number of classes.

    Returns:
        tf.estimator.EstimatorSpec: A tensorflow object containing the results
            of training, testing, or prediction (depending on the mode), to be
            evaluated by the tf.estimator.Estimator object.
    """
    # Parse out labels if training or testing
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = features.pop('label')

    # Input layer
    input_layer = tf.reshape(features['Image'], [-1, 200, 200, 3])

    # Convolution layer 1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=25,
                             padding='same',
                             activation=tf.nn.elu)

    # Max Pool layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    # Convolution layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=25,
                             padding='same',
                             activation=tf.nn.elu)

    # Max pool layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 50 * 50 * 64])

    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.elu)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=params['n_classes'])

    predictions = {
        # Generate predictons (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'soft_tensor' to the graph. It is used by PREDICT and by the
        # 'logging_hook.
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    predictions['loss'] = tf.reduce_mean(loss, name='m_loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
