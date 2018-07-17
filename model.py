#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:26:40 2018

@author: papi
"""

# Importing libraries
import numpy as np
import tensorflow as tf
from scipy import misc
import os

tf.logging.set_verbosity(tf.logging.INFO)

tensor_list_ellipse = []
tensor_list_ellipse_test = []
tensor_list_spiral = []

# Path to the directory of galaxy images
elliptical_path = '/Users/papi/tensorflow-for-poets-2/tf_files/otherdata/100w/Elliptical/train_elliptical'
elliptical_path_test = '/Users/papi/tensorflow-for-poets-2/tf_files/otherdata/100w/Elliptical/test_elliptical'
spiral_path = '/Users/papi/tensorflow-for-poets-2/tf_files/otherdata/100w/Spiral'

for file in os.listdir(elliptical_path):
    tensor = misc.imread(os.path.join(elliptical_path, file))
    tensor_list_ellipse.append(tensor)

for file in os.listdir(elliptical_path_test):
    tensor3 = misc.imread(os.path.join(elliptical_path_test, file))
    tensor_list_ellipse_test.append(tensor3)

tensor_train_ellipse = np.array(tensor_list_ellipse)*1.00
ellipse_train_labels = np.array([0]*len(tensor_train_ellipse))

tensor_test_ellipse = np.array(tensor_list_ellipse)*1.00
ellipse_test_labels = np.array([0]*len(tensor_test_ellipse))


def cnn_model(features, labels, mode):
    """CNN model function"""
    # Input layer
    input_layer = tf.reshape(features['x'], [-1, 200, 200, 3])

    # Convolution layer 1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=25,
                             padding='same',
                             activation=tf.nn.relu)

    # Max Pool layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    # Convolution layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=25,
                             padding='same',
                             activation=tf.nn.relu)

    # Max pool layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 50*50*64])
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    
    #Logits Layer
    logits = tf.layers.dense(inputs = dropout, units= 2)
    
    predictions = {
            #Generate predictons (for PREDICT and EVAL mode)
            "classes" : tf.argmax(input=logits, axis=1),
            #Add 'soft_tensor' to the graph. It is used by PREDICT and by the
            #'logging_book.
            'probabilities' : tf.nn.softmax(logits, name= 'softmax_tensor')
            }
    
    if mode  == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode= mode, predictions = predictions)
    
    #Configure Loss (for both TRAIN and EVAL modes)
#    onehot_labels = tf.one_hot(indices= [0,1], depth=2)
#    print('These are :', onehot_labels)
#    loss = tf.losses.softmax_cross_entropy(onehot_labels= onehot_labels, logits= logits)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    
    #Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.001)
        train_op = optimizer.minimize(
                loss= loss,
                global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op= train_op)
    
    #Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            'accuracy' : tf.metrics.accuracy(
                    labels= labels, predictions= predictions['classes'])}

    return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  train_data = tensor_train_ellipse # Returns np.array
  train_labels = np.asarray(ellipse_train_labels, dtype=np.int32)
  eval_data = tensor_test_ellipse # Returns np.array
  eval_labels = np.asarray(ellipse_test_labels, dtype=np.int32)
  #Create estimator
  classifier = tf.estimator.Estimator(
        model_fn=cnn_model)
  
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  classifier.train(
    input_fn=train_input_fn,
    steps=300,
    hooks=[logging_hook])
  
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
#  print(eval_results)
#  print(onehot_labels)
  # Evaluate the model and print results

  
if __name__ == '__main__':
    tf.app.run()