#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jean Donet

*** TO BE USED IN HPC *** (Paths are in HPC)

"""

#Importing libraries
import numpy      as np
import tensorflow as tf
import imageio    as io
import os

tf.logging.set_verbosity(tf.logging.INFO)

tensor_list_ellipse      = []                     #List to store image tensors for training
tensor_list_ellipse_test = []                     #List to store image tensors for testing

#Path to the directory of galaxy images
elliptical_path      = '/home/jdonet/train_elliptical'
elliptical_path_test = '/home/jdonet/test_elliptical'

for file in os.listdir(elliptical_path):
    tensor = io.imread(os.path.join(elliptical_path, file))        #Reading in training image as tensor
    tensor = tensor.astype(np.float32)                             #Casting tensor values to needed format
    
    tensor_list_ellipse.append(tensor)                             #Adding tensor to training list

for file in os.listdir(elliptical_path_test):
    tensor3 = io.imread(os.path.join(elliptical_path_test, file))  #Reading in testing image as tensor
    tensor3 = tensor3.astype(np.float32)                           #Casting tensor values to needed formnat
    
    tensor_list_ellipse_test.append(tensor3)                       #Adding tensor to testing list

tensor_train_ellipse = np.array(tensor_list_ellipse)               #Casting training list to numpy array
tensor_test_ellipse  = np.array(tensor_list_ellipse)               #Casting testing list to numpy array
ellipse_train_labels = np.array([0]*len(tensor_train_ellipse))     #Creating labels lists
ellipse_test_labels  = np.array([0]*len(tensor_test_ellipse))


def cnn_model(features, labels, mode):
    """CNN model function"""
    
    input_layer = tf.reshape(features['x'], [-1, 200, 200 , 3])              #Input layer with image dimensions
    
    
    conv1 = tf.layers.conv2d(                                                #Convolution layer 1
            inputs = input_layer,                                            #Selecting input layer as input for convolution
            filters = 32,                                                    #Specifying number of filters to be generated (32)
            kernel_size = 25,                                                #Dimensions of the convolution layer
            padding = 'same',                                                #Specifying zero padding     
            activation= tf.nn.relu)                                          #Soecifying ReLU activation layer
    
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size= 2, strides=2)   #Max Pooling layer with stride of 2
                                                                             #Takes in the conv 1 layer
   
    
    conv2 = tf.layers.conv2d(                                                #Convolution layer 2
            inputs = pool1,                                                  #Selecting input layer as input for convolution
            filters = 64,                                                    #Specifying number of filters to be generated (64)
            kernel_size = 25,                                                #Dimensions of the convolution layer
            padding = 'same',                                                #Specifying zero padding     
            activation= tf.nn.relu)                                          #Soecifying ReLU activation layer
    
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size= 2, strides=2)   #Max Pooling layer with stride of 2
                                                                             #Takes in the conv 2 layer
    
    
    pool2_flat  = tf.reshape(pool2, [-1,50*50*64])                                        #Reshaping the previous pool layer
    dense       = tf.layers.dense(inputs=pool2_flat, units=1024, activation= tf.nn.relu)  #Generating dense layer with 1024 neurons
    dropout     = tf.layers.dropout(
                  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)   #Dropout fn to prevent overfitting
    

    logits      = tf.layers.dense(inputs = dropout, units= 2)                     #Final CNN layer, with 2 output neurons for now
    
    predictions = {
            
            "classes" : tf.argmax(input=logits, axis=1),                          #Generate predictons (for PREDICT and EVAL mode)
            'probabilities' : tf.nn.softmax(logits, name= 'softmax_tensor')       #Softmax outputs for the logits layer
            }
    
    if mode  == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode= mode, predictions = predictions)  #Initializing PREDICT mode
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)   #Initializing cross entropy loss fn
    
    
    if mode == tf.estimator.ModeKeys.TRAIN:                                       #Configuring TRAIN mode
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.001)       #Selecting Stochastic Gradient Decent optimizer
        train_op = optimizer.minimize(
                loss= loss,                                                       #Minimizing error by implementing loss fn
                global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op= train_op)
    
    eval_metric_ops = {                                                           #Configuring evaluation metrics
            'accuracy' : tf.metrics.accuracy(                                     #Passing in labels for output
                    labels= labels, predictions= predictions['classes'])}         #Passing in the classes from logits layer

    return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  train_data     = tensor_train_ellipse                                #Setting training data as inout for CNN
  train_labels   = np.asarray(ellipse_train_labels, dtype=np.int32)    #Setting training label list for output 
  eval_data      = tensor_test_ellipse                                 #Setting testing data for evaluation of training
  eval_labels    = np.asarray(ellipse_test_labels, dtype=np.int32)     #Setting testing label list for output
  
  classifier     = tf.estimator.Estimator(
                   model_fn=cnn_model)                                 #Setting our CNN as the classifier


  tensors_to_log = {"probabilities": "softmax_tensor"}                 #Setting up logging for predictions
  logging_hook   = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=50)
  
  train_input_fn = tf.estimator.inputs.numpy_input_fn(                 #Train the model
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  classifier.train(
    input_fn=train_input_fn,
    steps=300,
    hooks=[logging_hook])
   
  eval_input_fn  = tf.estimator.inputs.numpy_input_fn(                 #Evaluate the model and print results
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)


  
if __name__ == '__main__':
    tf.app.run()