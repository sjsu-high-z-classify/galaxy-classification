#!/usr/bin/env python
# coding: utf-8

import os
from time import sleep

import numpy as np
import pandas as pd

import SciServer
from SciServer import SkyServer, Authentication

import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm

try:
    os.makedirs('data/img/')
except FileExistsError:
    pass

token = Authentication.login('jacaseyclyde', 'ub3rl337')

sql = 'SELECT gz2.specobjid, gz2.ra, gz2.dec, dr7.petroR90_r, ' \
      'gz2.t01_smooth_or_features_a01_smooth_weighted_fraction AS t01a01, ' \
      'gz2.t01_smooth_or_features_a02_features_or_disk_weighted_fraction AS t01a02, ' \
      'gz2.t01_smooth_or_features_a03_star_or_artifact_weighted_fraction AS t01a03, ' \
      'gz2.t02_edgeon_a04_yes_weighted_fraction AS t02a04, ' \
      'gz2.t02_edgeon_a05_no_weighted_fraction AS t02a05, ' \
      'gz2.t03_bar_a06_bar_weighted_fraction AS t03a06, ' \
      'gz2.t03_bar_a07_no_bar_weighted_fraction AS t03a07, ' \
      'gz2.t04_spiral_a08_spiral_weighted_fraction AS t04a08, ' \
      'gz2.t04_spiral_a09_no_spiral_weighted_fraction AS t04a09, ' \
      'gz2.t05_bulge_prominence_a10_no_bulge_weighted_fraction AS t05a10, ' \
      'gz2.t05_bulge_prominence_a11_just_noticeable_weighted_fraction AS t05a11, ' \
      'gz2.t05_bulge_prominence_a12_obvious_weighted_fraction AS t05a12, ' \
      'gz2.t05_bulge_prominence_a13_dominant_weighted_fraction AS t05a13, ' \
      'gz2.t06_odd_a14_yes_weighted_fraction AS t06a14, ' \
      'gz2.t06_odd_a15_no_weighted_fraction AS t06a15, ' \
      'gz2.t07_rounded_a16_completely_round_weighted_fraction AS t07a16, ' \
      'gz2.t07_rounded_a17_in_between_weighted_fraction AS t07a17, ' \
      'gz2.t07_rounded_a18_cigar_shaped_weighted_fraction AS t07a18, ' \
      'gz2.t08_odd_feature_a19_ring_weighted_fraction AS t08a19, ' \
      'gz2.t08_odd_feature_a20_lens_or_arc_weighted_fraction AS t08a20, ' \
      'gz2.t08_odd_feature_a21_disturbed_weighted_fraction AS t08a21, ' \
      'gz2.t08_odd_feature_a22_irregular_weighted_fraction AS t08a22, ' \
      'gz2.t08_odd_feature_a23_other_weighted_fraction AS t08a23, ' \
      'gz2.t08_odd_feature_a24_merger_weighted_fraction AS t08a24, ' \
      'gz2.t08_odd_feature_a38_dust_lane_weighted_fraction AS t08a38, ' \
      'gz2.t09_bulge_shape_a25_rounded_weighted_fraction AS t09a25, ' \
      'gz2.t09_bulge_shape_a26_boxy_weighted_fraction AS t09a26, ' \
      'gz2.t09_bulge_shape_a27_no_bulge_weighted_fraction AS t09a27, ' \
      'gz2.t10_arms_winding_a28_tight_weighted_fraction AS t10a28, ' \
      'gz2.t10_arms_winding_a29_medium_weighted_fraction AS t10a29, ' \
      'gz2.t10_arms_winding_a30_loose_weighted_fraction AS t10a30, ' \
      'gz2.t11_arms_number_a31_1_weighted_fraction AS t11a31, ' \
      'gz2.t11_arms_number_a32_2_weighted_fraction AS t11a32, ' \
      'gz2.t11_arms_number_a33_3_weighted_fraction AS t11a33, ' \
      'gz2.t11_arms_number_a34_4_weighted_fraction AS t11a34, ' \
      'gz2.t11_arms_number_a36_more_than_4_weighted_fraction AS t11a36, ' \
      'gz2.t11_arms_number_a37_cant_tell_weighted_fraction AS t11a37 ' \
      'FROM zoo2MainSpecz as gz2 JOIN SpecDR7 AS dr7 ' \
      'ON gz2.dr7objid=dr7.dr7objid'

try:
    gz2 = pd.read_hdf('data/gz2.h5')
except:
    gz2 = SkyServer.sqlSearch(sql=sql, dataRelease='DR15')
    gz2 = gz2.assign(imgpath=None)
    gz2.to_hdf('data/gz2.h5', key='gz2', append=False, mode='w')

with tqdm(total=len(gz2.index), unit='object') as pbar:
    for index, obj in gz2.iterrows():
        if obj['imgpath'] is not None:
            pbar.update(1)
            continue
        
        attempts = 0
        img = None
        while True:
            try:
                img = SkyServer.getJpegImgCutout(ra=obj['ra'], dec=obj['dec'],
                                                 width=424, height=424, scale=(0.02 * obj['petroR90_r']),
                                                 dataRelease='DR15')
            except Exception as e:
                attempts += 1
                if attempts >= 5:
                    break
                sleep(1)
                continue
            break
        
        if img is not None:
            imgpath = "data/img/{0}.jpeg".format(obj['specobjid'])
            Image.fromarray(img).save(imgpath)
            gz2.at[index, 'imgpath'] = imgpath
            gz2.to_hdf('data/gz2.h5', key='gz2', append=False, mode='w')
            
        pbar.update(1)

gz2.dropna(axis=0, subset=['imgpath'], inplace=True)
gz2.to_hdf('data/gz2.h5', key='gz2', append=False, mode='w')



