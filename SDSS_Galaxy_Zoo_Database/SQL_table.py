# In[1]:

import pandas as pd
import numpy as np
import mechanicalsoup  as ms
from SciServer import Authentication, CasJobs
import time

# In[2]:

Authentication_loginName = '***'
Authentication_loginPassword = '***'
token = Authentication.login(Authentication_loginName, Authentication_loginPassword)

# In[3]:

def database(n):
    query = "SELECT TOP {} specobjid, objid as objid8, ra, dec, spiral, elliptical, uncertain FROM ZooSpec".format(n)
    responseStream = CasJobs.executeQuery(query, "DR14", format="pandas")
    
    data = pd.DataFrame(responseStream)
#     print("\n---Query---\n{}\n---Result---\n{}".format(query, responseStream))
    data['Gtype'] = ''
    
    for i in range(np.size(data.specobjid)):
        if data.spiral[i] == 1:
            data.loc[i,'Gtype'] = 'S'
        elif data.elliptical[i] == 1:
            data.loc[i,'Gtype'] = 'E'
        elif data.uncertain[i] == 1:
            data.loc[i,'Gtype'] = 'UN'
            
    data = data.drop(columns=['spiral', 'elliptical','uncertain'])
    data.to_csv('Skyserver_SQL_{}.csv'.format(time.strftime("%Y-%m-%d_%H_%M_%S")))
    return database

# In[4]:

database(1000)