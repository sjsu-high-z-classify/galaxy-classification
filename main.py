#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:52:11 2018

@author: jacaseyclyde
"""

galData = pd.read_csv('Book.csv')
galData = pd.DataFrame(galData)
    
# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((galData.RA,
                                              galData.DEC,
                                              galData.TYPE))
dataset = dataset.map(distorted_inputs)