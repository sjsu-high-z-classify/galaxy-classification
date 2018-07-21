#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database query and catalogue population module

This module is for populating a csv catalogue of classified galaxies for use in
training the Convolutional Neural Network (CNN) contained in the rest of this
project.

Technical info:
    Source Catalogue: Galaxy Zoo 1

@author: Hiren Thummar
"""
import pandas as pd
import numpy as np
from SciServer import Authentication, CasJobs


def dataquery(records, username, password):
    """
    Queries CasJobs to retrieve Galaxy Zoo 1 catalogue

    Args:
        records -- number of records to retreive
    """

    token = Authentication.login(username, password)

    query = 'SELECT TOP {0}'.format(records) + ' specobjid, objid as objid8,' \
        + ' ra, dec, spiral, elliptical, uncertain FROM ZooSpec'
    response_stream = CasJobs.executeQuery(query, 'DR14', format='pandas')

    data = pd.DataFrame(response_stream)
    data['Gtype'] = ''

    for i in range(np.size(data.specobjid)):
        if data.spiral[i] == 1:
            data.loc[i, 'Gtype'] = 'S'
        elif data.elliptical[i] == 1:
            data.loc[i, 'Gtype'] = 'E'
        elif data.uncertain[i] == 1:
            data.loc[i, 'Gtype'] = 'UN'

    data = data.drop(columns=['spiral', 'elliptical', 'uncertain'])
    data.to_csv('catalogue.csv')
