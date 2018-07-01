
# coding: utf-8

# In[10]:


import mechanicalsoup  as ms


# In[11]:


url = "http://skyserver.sdss.org/dr14/en/tools/search/sql.aspx"
br = ms.StatefulBrowser()


# In[12]:


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
    
def writer(name, data):
    # writes data to a file
    f = open(name, 'w')
    f.write(data)
    f.close()
    return writer


# In[13]:


# this one uses table GALAXY ZOO 1 from SDSS DR14
n = 10000
s = "SELECT TOP {}         specobjid, objid as objid8, ra, dec,         spiral, elliptical, uncertain     FROM ZooSpec".format(n)

# print(s)

SDSS = SDSS_select(s)

# print(SDSS)
filename = 'data_GZ1.csv'
writer(filename, SDSS)


# In[14]:


# n = 10000
# s = "SELECT TOP {}         specobjid, dr8objid, ra, dec, t01_smooth_or_features_a01_smooth_flag,         t01_smooth_or_features_a02_features_or_disk_flag, t01_smooth_or_features_a03_star_or_artifact_flag,         t02_edgeon_a04_yes_flag, t02_edgeon_a05_no_flag, t03_bar_a06_bar_flag, t03_bar_a07_no_bar_flag,         t04_spiral_a08_spiral_flag, t04_spiral_a09_no_spiral_flag, t05_bulge_prominence_a10_no_bulge_flag,         t05_bulge_prominence_a11_just_noticeable_flag, t05_bulge_prominence_a12_obvious_flag,         t05_bulge_prominence_a13_dominant_flag, t06_odd_a14_yes_flag, t06_odd_a15_no_flag,         t07_rounded_a16_completely_round_flag, t07_rounded_a17_in_between_flag, t07_rounded_a18_cigar_shaped_flag,         t08_odd_feature_a19_ring_flag, t08_odd_feature_a20_lens_or_arc_flag, t08_odd_feature_a21_disturbed_flag,         t08_odd_feature_a22_irregular_flag, t08_odd_feature_a23_other_flag, t08_odd_feature_a24_merger_flag,         t08_odd_feature_a38_dust_lane_flag, t09_bulge_shape_a25_rounded_flag, t09_bulge_shape_a26_boxy_flag,          t09_bulge_shape_a27_no_bulge_flag, t10_arms_winding_a28_tight_flag, t10_arms_winding_a29_medium_flag,         t10_arms_winding_a30_loose_flag, t11_arms_number_a31_1_flag, t11_arms_number_a32_2_flag,         t11_arms_number_a33_3_flag, t11_arms_number_a34_4_flag, t11_arms_number_a36_more_than_4_flag,         t11_arms_number_a37_cant_tell_flag     FROM zoo2MainSpecz".format(n)
# print(s)

# SDSS = SDSS_select(s)

# print(SDSS)
# filename = 'data_GZ2.csv'
# writer(filename, SDSS)

