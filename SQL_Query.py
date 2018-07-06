
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import mechanicalsoup  as ms
import time


# In[2]:


url = "http://skyserver.sdss.org/dr14/en/tools/search/sql.aspx"
br = ms.StatefulBrowser()


# In[3]:


def SDSS_select(sql):
    '''
    input:  string with a valid SQL query
    output: csv
    source: http://balbuceosastropy.blogspot.com/2013/10/an-easy-way-to-make-sql-queries-from.html
    '''
    br.open(url)
    br.select_form("[name=sql]")
    br['cmd'] = sql
    br["format"]="csv"
    response = br.submit_selected()
    return response.text
    
def writer(filename, data):
    # writes data to a file
    f = open(filename, 'w')
    f.write(data)
    f.close()
    return writer

def database(n):
    s = "SELECT TOP {}         specobjid, objid as objid8, ra, dec,         spiral, elliptical, uncertain     FROM ZooSpec".format(n)

    SDSS = SDSS_select(s)
    writer('data_GZ1.csv', SDSS)
    
    data = pd.read_csv('data_GZ1.csv',header = 1)
    data = pd.DataFrame(data)
    
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


# In[8]:


database(1000)

