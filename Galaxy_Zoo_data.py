import pandas as pd
import urllib.request

n = 2  # number of files to download
data = pd.read_csv('Book.csv')
data = pd.DataFrame(data)

data.head()

for i in range(n):#np.size(data.OBJID)):
    urllib.request.urlretrieve('http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={0}&dec={1}&scale=.2&width=200&height=200'.format(data.RA[i], data.DEC[i]),'Img_{0}_{1}.jpeg'.format(i,data.OBJID[i]))

