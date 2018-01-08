# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 11:42:31 2017
This file analyzes the texts of the tweets to find the bus ids and bus route numbers
@author: SAA
"""


import pandas as pd
import pygmaps 
import numpy as np
#import scipy.optimize as least_square

data = pd.read_csv("25Nov17-translink.csv",sep=',')

data = data[pd.notnull(data['user'])]

data = data.drop_duplicates()

data.columns = [u'time',u'user', u'text', u'latitude', u'longitude']

data.index = range(len(data))

data2 = pd.read_csv("saved_data7.csv",sep=',')

data2 = data2[pd.notnull(data2['user'])]

data2 = data2.drop_duplicates()

data2.columns = [u'time',u'user', u'text', u'latitude', u'longitude']

data2.index = range(len(data2))


data3 = pd.read_csv("saved_data9.csv",sep=',')

data3 = data3[pd.notnull(data3['user'])]

data3 = data3.drop_duplicates()

data3.columns = [u'time',u'user', u'text', u'latitude', u'longitude']

data3.index = range(len(data3))


data=pd.concat([data3,data2,data], ignore_index=True)
data = data.drop_duplicates()

latz = 49.1907185	# geographical centre of search
longz = -122.90168	# geographical centre of search
zoom = 10

mymap = pygmaps.maps(latz, longz, zoom)

for indx in data.loc[data['latitude']!=0].index:
    mymap.addpoint(data.loc[indx]['latitude'],data.loc[indx]['longitude'],"#D3D3D3",data.loc[indx]['text'])

#mymap.draw('./translink.html')
numberlst = np.array([])

for indx in data.index:
    text = data.loc[indx]['text'].replace('#',' ').replace(',',' ')
    numbers = [int(s) for s in text.split() if s.isdigit()]
    if numbers: numberlst = np.append(numberlst,numbers)
    
stopidlst = numberlst[(100000>numberlst) & (numberlst>10000)]


#############################################################################
##                                                                         ##
##                           stop ids matching                             ##
##                                                                         ##
#############################################################################


data3 = pd.read_csv("stopsgtfs.csv",sep=',')

data3 = data3.drop_duplicates()

data3.columns = [u'id1',u'id2', u'id',u'address 1',u'address 2', u'latitude', u'longitude','','','','','']

data3.index = range(len(data3))

data3['repetition']=pd.Series(np.zeros(len(data3)),index=data3.index)

for stpid in stopidlst:
    data3.loc[data3.id==str(int(stpid)),'repetition']+=1
              

## colour scaling
## colour scaling function
def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b  

minrgb = data3['repetition'].min()
maxrgb = data3['repetition'].max()
          
for indx in data3[data3['repetition']!=0].index:
    mymap.addpoint(data3.loc[indx,'latitude'],data3.loc[indx,'longitude'],'#%02x%02x%02x' % rgb(minrgb, maxrgb,data3.loc[indx,'repetition']))

#mymap.draw('./translink.html')



#############################################################################
##                                                                         ##
##                           route ids matching                            ##
##                                                                         ##
#############################################################################

data4 = pd.read_csv("sa_surrey_routes.csv",sep=',')

data4.columns = [u'id',u'station', u'route']

data4 = data4.drop_duplicates('route')

data4.index = range(len(data4))

data4['repetition']=pd.Series(np.zeros(len(data4)),index=data4.index)

for number in numberlst:
    data4.loc[data4.route==str(int(number)),'repetition']+=1

data4.sort('repetition',ascending=0).head()







