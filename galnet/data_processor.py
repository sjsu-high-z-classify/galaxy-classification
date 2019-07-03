#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


INPUT_DIM = (69, 69)
DATA_FORMAT = 'channels_last'
BATCH_SIZE = 256

gz2 = pd.read_hdf('data/gz2.h5')

def entropy(p_i):
    logp_i = np.log(p_i)
    logp_i[logp_i == -np.inf] = 1.
    return -np.sum(p_i * logp_i)


def agreement(row):
    t01 = np.array([row['t01a01'], row['t01a02'], row['t01a03']])
    t02 = np.array([row['t02a04'], row['t02a05']])
    t03 = np.array([row['t03a06'], row['t03a07']])
    t04 = np.array([row['t04a08'], row['t04a09']])
    t05 = np.array([row['t05a10'], row['t05a11'], row['t05a12'], row['t05a13']])
    t06 = np.array([row['t06a14'], row['t06a15']])
    t07 = np.array([row['t07a16'], row['t07a17'], row['t07a18']])
    t08 = np.array([row['t08a19'], row['t08a20'], row['t08a21'], row['t08a22'], row['t08a23'], row['t08a24'], row['t08a38']])
    t09 = np.array([row['t09a25'], row['t09a26'], row['t09a27']])
    t10 = np.array([row['t10a28'], row['t10a29'], row['t10a30']])
    t11 = np.array([row['t11a31'], row['t11a32'], row['t11a33'], row['t11a34'], row['t11a36'], row['t11a37']])
    return (1 - entropy(t01) / np.log(len(t01)),
            1 - entropy(t02) / np.log(len(t02)),
            1 - entropy(t03) / np.log(len(t03)),
            1 - entropy(t04) / np.log(len(t04)),
            1 - entropy(t05) / np.log(len(t05)),
            1 - entropy(t06) / np.log(len(t06)),
            1 - entropy(t07) / np.log(len(t07)),
            1 - entropy(t08) / np.log(len(t08)),
            1 - entropy(t09) / np.log(len(t09)),
            1 - entropy(t10) / np.log(len(t10)),
            1 - entropy(t11) / np.log(len(t11)))


def one_hot_encoder(row):
    t01 = np.zeros(3)
    t02 = np.zeros(2)
    t03 = np.zeros(2)
    t04 = np.zeros(2)
    t05 = np.zeros(4)
    t06 = np.zeros(2)
    t07 = np.zeros(3)
    t08 = np.zeros(7)
    t09 = np.zeros(3)
    t10 = np.zeros(3)
    t11 = np.zeros(6)
    
    # assign the answer code that corresponds to the highest vote fraction
    t01[np.argmax([row['t01a01'], row['t01a02'], row['t01a03']])] = 1
    t02[np.argmax([row['t02a04'], row['t02a05']])] = 1
    t03[np.argmax([row['t03a06'], row['t03a07']])] = 1
    t04[np.argmax([row['t04a08'], row['t04a09']])] = 1
    t05[np.argmax([row['t05a10'], row['t05a11'], row['t05a12'], row['t05a13']])] = 1
    t06[np.argmax([row['t06a14'], row['t06a15']])] = 1
    t07[np.argmax([row['t07a16'], row['t07a17'], row['t07a18']])] = 1
    t08[np.argmax([row['t08a19'], row['t08a20'], row['t08a21'], row['t08a22'], row['t08a23'], row['t08a24'], row['t08a38']])] = 1
    t09[np.argmax([row['t09a25'], row['t09a26'], row['t09a27']])] = 1
    t10[np.argmax([row['t10a28'], row['t10a29'], row['t10a30']])] = 1
    t11[np.argmax([row['t11a31'], row['t11a32'], row['t11a33'], row['t11a34'], row['t11a36'], row['t11a37']])] = 1
    
    return (t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11)


(gz2['a01'], gz2['a02'], gz2['a03'],
 gz2['a04'], gz2['a05'], gz2['a06'],
 gz2['a07'], gz2['a08'], gz2['a09'],
 gz2['a10'], gz2['a11']) = zip(*gz2.apply(lambda row: agreement(row), axis=1))

(gz2['t01'], gz2['t02'], gz2['t03'],
 gz2['t04'], gz2['t05'], gz2['t06'],
 gz2['t07'], gz2['t08'], gz2['t09'],
 gz2['t10'], gz2['t11']) = zip(*gz2.apply(lambda row: one_hot_encoder(row), axis=1))

gz2.to_hdf('data/gz2.h5', key='gz2', append=False, mode='w')