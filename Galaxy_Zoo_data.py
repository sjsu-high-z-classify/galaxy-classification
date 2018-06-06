
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import astropy as ast
import urllib.request

import ra_dec_conv as rd


# In[5]:


data = pd.read_csv('Book.csv')
data = pd.DataFrame(data);


# In[6]:


data.loc[data['OBJID'] == 587730775499407000]

#data.head()


# data['ra_d'] = None
# data['dec_d'] = None
# for i in range(200000):#np.size(data.OBJID)):
#     [data.ra_d[i], data.dec_d[i]] = rd.ra_dec_conv(data.RA[i], data.DEC[i])
# 

# In[15]:


cd ~/Documents/GitHub/Infinity_Categorizer/test


# In[5]:


for i in range(2):#00000):#np.size(data.OBJID)):
    urllib.request.urlretrieve('http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={0}&dec={1}&scale=.2&width=200&height=200'.format(data.RA[i], data.DEC[i]),'Img_{}_{}.jpeg'.format(i,data.OBJID[i]))


# In[8]:


get_ipython().run_line_magic('pinfo', 'rd.ra_dec_conv')


# In[16]:


# r = 0
for i in range(200):
    if data.SPIRAL[i] == 1:
        urllib.request.urlretrieve('http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={0}&dec={1}&scale=.2&width=200&height=200'.format(data.RA[i], data.DEC[i]),'Img_{}_{}.jpeg'.format(i,data.OBJID[i]))
#         print('{}_{}_True'.format(i,data.OBJID[r]))
#         r = r+1
#     else:
#         print('{}_{}_False'.format(i,data.OBJID[r]))
#         r = r+1

