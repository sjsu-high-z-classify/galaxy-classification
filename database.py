#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN model definition module.
Copyright (C) 2018  Hiren Thummar and J. Andrew Casey-Clyde

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

Todo:
    * Update to GalaxyZoo2
"""

import pandas as pd
import numpy as np
from SciServer import Authentication, CasJobs


def dataquery(records, username, password):
    """Download galaxy classification catalogue.

    Downloads a catalogue of specobjid, objid8, ra, dec, and classification
    from the GalaxyZoo catalogue

    Args:
        records (int): Number of records to retreive.
        username (str): CasJobs username.
        password (str): CasJobs password.
    """
    Authentication.login(username, password)

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
