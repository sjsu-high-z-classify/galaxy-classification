#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:52:11 2018

@author: jacaseyclyde
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pipeline

import pandas as pd


def main():
    gal_data = pd.read_csv('Book.csv')
    gal_data = pd.DataFrame(gal_data)

    dataset = pipeline.train_input_fn(gal_data, 100)

    return dataset


if __name__ == '__main__':
    dataset = main()
