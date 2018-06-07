
# coding: utf-8

# In[5]:


import pandas as pd
import urllib.request


# In[7]:


n = 2 #number of files to download
data = pd.read_csv('Book.csv')
data = pd.DataFrame(data);


# In[8]:


data.head()


# In[9]:


cd ~/Documents/GitHub/Infinity_Categorizer


# In[11]:


for i in range(2):#np.size(data.OBJID)):
    urllib.request.urlretrieve('http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={0}&dec={1}&scale=.2&width=200&height=200'.format(data.RA[i], data.DEC[i]),'Img_{0}_{1}.jpeg'.format(i,data.OBJID[i]))

