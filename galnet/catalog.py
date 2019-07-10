#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module to build catalog of GZ2 galaxies for use with GalNet.
# Copyright (C) 2018-2019  J. Andrew Casey-Clyde and Hiren Thummar.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import argparse

import numpy as np
import pandas as pd

from SciServer import SkyServer, Authentication

from PIL import Image

from tqdm import tqdm

CATALOG = None
TOKEN = None


def _authenticate():
    global TOKEN
    global PARSER
    TOKEN = Authentication.login(PARSER.USERNAME, PARSER.PASSWORD)
    return TOKEN


def _entropy(p_i):
    logp_i = np.log(p_i)
    logp_i[logp_i == -np.inf] = 1.
    return -np.sum(p_i * logp_i)


def agreement(row):
    """Calculates GZ2 agreement.

    Quantifies the level of agreement voters had on each question for a
    GalaxyZoo2 galaxy.

    """
    t01 = np.array([row['t01a01'], row['t01a02'], row['t01a03']])
    t02 = np.array([row['t02a04'], row['t02a05']])
    t03 = np.array([row['t03a06'], row['t03a07']])
    t04 = np.array([row['t04a08'], row['t04a09']])
    t05 = np.array([row['t05a10'], row['t05a11'], row['t05a12'],
                    row['t05a13']])
    t06 = np.array([row['t06a14'], row['t06a15']])
    t07 = np.array([row['t07a16'], row['t07a17'], row['t07a18']])
    t08 = np.array([row['t08a19'], row['t08a20'], row['t08a21'], row['t08a22'],
                    row['t08a23'], row['t08a24'], row['t08a38']])
    t09 = np.array([row['t09a25'], row['t09a26'], row['t09a27']])
    t10 = np.array([row['t10a28'], row['t10a29'], row['t10a30']])
    t11 = np.array([row['t11a31'], row['t11a32'], row['t11a33'], row['t11a34'],
                    row['t11a36'], row['t11a37']])
    return (1 - _entropy(t01) / np.log(len(t01)),
            1 - _entropy(t02) / np.log(len(t02)),
            1 - _entropy(t03) / np.log(len(t03)),
            1 - _entropy(t04) / np.log(len(t04)),
            1 - _entropy(t05) / np.log(len(t05)),
            1 - _entropy(t06) / np.log(len(t06)),
            1 - _entropy(t07) / np.log(len(t07)),
            1 - _entropy(t08) / np.log(len(t08)),
            1 - _entropy(t09) / np.log(len(t09)),
            1 - _entropy(t10) / np.log(len(t10)),
            1 - _entropy(t11) / np.log(len(t11)))


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
    t05[np.argmax([row['t05a10'], row['t05a11'], row['t05a12'],
                   row['t05a13']])] = 1
    t06[np.argmax([row['t06a14'], row['t06a15']])] = 1
    t07[np.argmax([row['t07a16'], row['t07a17'], row['t07a18']])] = 1
    t08[np.argmax([row['t08a19'], row['t08a20'], row['t08a21'], row['t08a22'],
                   row['t08a23'], row['t08a24'], row['t08a38']])] = 1
    t09[np.argmax([row['t09a25'], row['t09a26'], row['t09a27']])] = 1
    t10[np.argmax([row['t10a28'], row['t10a29'], row['t10a30']])] = 1
    t11[np.argmax([row['t11a31'], row['t11a32'], row['t11a33'], row['t11a34'],
                   row['t11a36'], row['t11a37']])] = 1

    return (t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11)


def crop_images(df, w, h):
    """Crops all images in dataframe to central pixels.

    Crops images in dataframe to their to central `w` by `h` pixels. If
    an even crop cannot be found, one less pixel will be removed from
    the left and/or top.

    """
    paths = np.unique(df['imgpath'])
    with tqdm(total=len(paths), unit='object') as pbar:
        for path in paths:
            im = Image.open(path)

            left = (im.width - w) // 2
            right = left + w

            upper = (im.height - h) // 2
            lower = upper + h

            im = im.crop(box=(left, upper, right, lower))
            im.save(path)
            pbar.update()
    return


def get_catalog():
    """Downloads GZ2 object catalog.

    Retrieve a dataframe of the objects present in the GalaxyZoo2
    catalog, as well as weighted fraction task answers for each object.

    """
    if TOKEN is None:
        _authenticate()

    sql = 'SELECT gz2.specobjid, gz2.ra, gz2.dec, dr7.petroR90_r, ' \
          'gz2.t01_smooth_or_features_a01_smooth_weighted_fraction' \
          ' AS t01a01, ' \
          'gz2.t01_smooth_or_features_a02_features_or_disk_weighted_fraction' \
          ' AS t01a02, ' \
          'gz2.t01_smooth_or_features_a03_star_or_artifact_weighted_fraction' \
          ' AS t01a03, ' \
          'gz2.t02_edgeon_a04_yes_weighted_fraction' \
          ' AS t02a04, ' \
          'gz2.t02_edgeon_a05_no_weighted_fraction' \
          ' AS t02a05, ' \
          'gz2.t03_bar_a06_bar_weighted_fraction' \
          ' AS t03a06, ' \
          'gz2.t03_bar_a07_no_bar_weighted_fraction' \
          ' AS t03a07, ' \
          'gz2.t04_spiral_a08_spiral_weighted_fraction' \
          ' AS t04a08, ' \
          'gz2.t04_spiral_a09_no_spiral_weighted_fraction' \
          ' AS t04a09, ' \
          'gz2.t05_bulge_prominence_a10_no_bulge_weighted_fraction' \
          ' AS t05a10, ' \
          'gz2.t05_bulge_prominence_a11_just_noticeable_weighted_fraction' \
          ' AS t05a11, ' \
          'gz2.t05_bulge_prominence_a12_obvious_weighted_fraction' \
          ' AS t05a12, ' \
          'gz2.t05_bulge_prominence_a13_dominant_weighted_fraction' \
          ' AS t05a13, ' \
          'gz2.t06_odd_a14_yes_weighted_fraction' \
          ' AS t06a14, ' \
          'gz2.t06_odd_a15_no_weighted_fraction' \
          ' AS t06a15, ' \
          'gz2.t07_rounded_a16_completely_round_weighted_fraction' \
          ' AS t07a16, ' \
          'gz2.t07_rounded_a17_in_between_weighted_fraction' \
          ' AS t07a17, ' \
          'gz2.t07_rounded_a18_cigar_shaped_weighted_fraction' \
          ' AS t07a18, ' \
          'gz2.t08_odd_feature_a19_ring_weighted_fraction' \
          ' AS t08a19, ' \
          'gz2.t08_odd_feature_a20_lens_or_arc_weighted_fraction' \
          ' AS t08a20, ' \
          'gz2.t08_odd_feature_a21_disturbed_weighted_fraction' \
          ' AS t08a21, ' \
          'gz2.t08_odd_feature_a22_irregular_weighted_fraction' \
          ' AS t08a22, ' \
          'gz2.t08_odd_feature_a23_other_weighted_fraction' \
          ' AS t08a23, ' \
          'gz2.t08_odd_feature_a24_merger_weighted_fraction' \
          ' AS t08a24, ' \
          'gz2.t08_odd_feature_a38_dust_lane_weighted_fraction' \
          ' AS t08a38, ' \
          'gz2.t09_bulge_shape_a25_rounded_weighted_fraction' \
          ' AS t09a25, ' \
          'gz2.t09_bulge_shape_a26_boxy_weighted_fraction' \
          ' AS t09a26, ' \
          'gz2.t09_bulge_shape_a27_no_bulge_weighted_fraction' \
          ' AS t09a27, ' \
          'gz2.t10_arms_winding_a28_tight_weighted_fraction' \
          ' AS t10a28, ' \
          'gz2.t10_arms_winding_a29_medium_weighted_fraction' \
          ' AS t10a29, ' \
          'gz2.t10_arms_winding_a30_loose_weighted_fraction' \
          ' AS t10a30, ' \
          'gz2.t11_arms_number_a31_1_weighted_fraction' \
          ' AS t11a31, ' \
          'gz2.t11_arms_number_a32_2_weighted_fraction' \
          ' AS t11a32, ' \
          'gz2.t11_arms_number_a33_3_weighted_fraction' \
          ' AS t11a33, ' \
          'gz2.t11_arms_number_a34_4_weighted_fraction' \
          ' AS t11a34, ' \
          'gz2.t11_arms_number_a36_more_than_4_weighted_fraction' \
          ' AS t11a36, ' \
          'gz2.t11_arms_number_a37_cant_tell_weighted_fraction' \
          ' AS t11a37 ' \
          'FROM zoo2MainSpecz as gz2 JOIN SpecDR7 AS dr7 ' \
          'ON gz2.dr7objid=dr7.dr7objid'

    return SkyServer.sqlSearch(sql=sql, dataRelease='DR15')


def get_images(df, w=424, h=424, scale=.02,
               data_dir='data', img_dir='img', catalog_name='gz2'):
    """Get images for objects present in pandas dataframe.

    Downloads JPEG images of objects from SDSS to disk storage, naming
    images according to their :attr:`specobjid`. Filepaths are
    additionally stored in the dataframe, such that the dataframe can
    be used with :class:`keras.preprocessing.image.ImageDataGenerator`

    """
    catalog_path = os.path.join(data_dir, catalog_name + '.h5')
    df = df.assign(imgpath=None)
    with tqdm(total=len(df.index), unit='object') as pbar:
        for index, obj in df.iterrows():
            # allows for restarting incomplete downloads by
            # checking for all entries if the path already exists
            # XXX: tqdm includes better ways to approach this
            if obj['imgpath'] is not None:
                pbar.update(1)
                continue

            img = None
            img = SkyServer.getJpegImgCutout(ra=obj['ra'],
                                             dec=obj['dec'],
                                             width=w, height=h,
                                             scale=(scale * obj['petroR90_r']),
                                             dataRelease='DR15')

            if img is not None:
                imgpath = os.path.join(img_dir,
                                       '{0}.jpeg'.format(obj['specobjid']))
                Image.fromarray(img).save(imgpath)
                df.at[index, 'imgpath'] = imgpath
                df.to_hdf(catalog_path, key=catalog_name,
                          append=False, mode='w')

            pbar.update(1)

    return df.dropna(axis=0, subset=['imgpath'], inplace=True)


def build(data_path='data', image_path='img', catalog_name='gz2'):
    """Builds a catalog of galaxy images.

    Downloads a catalog of GZ2 objects, as well as associated jpeg
    images of each object. Catalog info is stored as an h5 archive.

    Parameters
    ----------

    """
    # create relevant paths
    catalog_path = os.path.join(data_path, catalog_name + '.h5')
    img_path = os.path.join(data_path, image_path)
    try:
        os.makedirs(img_path)
    except FileExistsError:
        pass

    # download and save an archive of GZ2 objects
    df = get_catalog()
    df.to_hdf(catalog_path, key='gz2', append=False, mode='w')

    # get the images we want from SDSS
    df = get_images(gz2, data_dir=data_path, img_dir=img_path,
                    catalog_name=catalog_name)
    df.to_hdf(catalog_path, key='gz2', append=False, mode='w')
    return df


def process(df, data_path='data', catalog_name='gz2'):
    catalog_path = os.path.join(data_path, catalog_name + '.h5')

    # calculate the agreement for each question/sample
    (df['a01'], df['a02'], df['a03'],
     df['a04'], df['a05'], df['a06'],
     df['a07'], df['a08'], df['a09'],
     df['a10'], df['a11']) = zip(*df.apply(lambda row: agreement(row),
                                           axis=1))
    df.to_hdf(catalog_path, key=catalog_name, append=False, mode='w')

    # onehot encode classifications
    (df['t01'], df['t02'], df['t03'],
     df['t04'], df['t05'], df['t06'],
     df['t07'], df['t08'], df['t09'],
     df['t10'], df['t11']) = zip(*df.apply(lambda row: one_hot_encoder(row),
                                           axis=1))
    df.to_hdf(catalog_path, key=catalog_name, append=False, mode='w')
    return df


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Build a catalogue "
                                     "of GZ2 images.")
#    PARSER.add_argument('-u', '--username', dest='USERNAME',
#                        help='SciServer username', default='', action='store',
#                        type=str, required=True)
#    PARSER.add_argument('-p', '--password', dest='PASSWORD',
#                        help='SciServer password',  default='', action='store',
#                        type=str, required=True)

    gz2 = pd.read_hdf('../data/gz2.h5')
    gz2 = process(gz2, data_path='../data')
    print("Processed successfully.")
