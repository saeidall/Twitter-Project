# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:51:16 2017

@author: SAA
"""

import pandas as pd
import pygmaps 
import numpy as np
import csv
#import scipy.optimize as least_square
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

#import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################

data = pd.read_csv("saved_data3and4.csv",sep=',')

data = data[pd.notnull(data['user'])]

data = data.drop_duplicates()

data.columns = [u'user', u'text', u'latitude', u'longitude', u'time']

data.index = range(len(data))

#######################################################################

data2 = pd.read_csv("saved_data6.csv",sep=',')

data2 = data2[pd.notnull(data2['user'])]

data2 = data2.drop_duplicates()

data2.columns = [u'user', u'text', u'latitude', u'longitude', u'time']

data2.index = range(len(data2))

#######################################################################

data3 = pd.read_csv("saved_data8.csv",sep=',')

data3 = data3[pd.notnull(data3['user'])]

data3 = data3.drop_duplicates()

data3.columns = [u'user', u'text', u'latitude', u'longitude', u'time']

data3.index = range(len(data3))

#######################################################################
# Putthing the data together
data=pd.concat([data3,data2,data], ignore_index=True)
data = data.drop_duplicates()

# defining function for color definitions and scaling
def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b  


# initialize
userlst = []
datelst = [[]]
pointlst = [[]]

for i in data.index:
    
    if not (data.loc[i]['user'] in userlst):
        
        userlst.append(data.loc[i]['user'])
        
        for j in data[data['user']==data.loc[i]['user']].index:
            datelst[-1].append(data.loc[j]['time'])
            pointlst[-1].append([data.loc[j]['longitude'],data.loc[j]['latitude']])
        datelst.append([])
        pointlst.append([])



latitude = 49.1907185	# geographical centre of search
longitude = -122.90168	# geographical centre of search
zoom = 10
        
mymap = pygmaps.maps(latitude, longitude, zoom)

# Creating the map of connections of people in Surrey, using their tweets with short travel distances
counter =0
surrey_points = []
for path_ in pointlst:
    if len(path_)>1:
        for i in range(1,len(path_)):
            if ((path_[i-1][0]-path_[i][0])**2.0+(path_[i-1][1]-path_[i][1])**2.0)**.5 <.15:
                if ((path_[i][0]<49.218415) and (path_[i][1]>-122.937306)) or ((path_[i-1][0]<49.218415) and (path_[i-1][1]>-122.937306)):
                    mymap.addpath([path_[i-1],path_[i]],"#00FF00",2)
                    surrey_points.append(path_[i-1])
                    surrey_points.append(path_[i])
                    counter += 1
surrey_points = pd.DataFrame(np.array(surrey_points)).drop_duplicates()
surrey_points.index = range(len(surrey_points))

k_means=KMeans(init='k-means++',n_clusters=30,n_init=200,max_iter=1000)
y=k_means.fit_predict(surrey_points)
cluster_centroids=k_means.cluster_centers_

minrgb = min(y)
maxrgb = max(y)

for indx in surrey_points.index:
    mymap.addpoint(surrey_points.loc[indx,0],surrey_points.loc[indx,1],'#%02x%02x%02x' % rgb(minrgb, maxrgb,y[indx]))

for cl in cluster_centroids:
    mymap.addradpoint(cl[0],cl[1],600)

mymap.draw('./mymap.html')

# Claculatin gthe connection demand in the Surrey based on the tweets
surreyconmap = pygmaps.maps(latitude, longitude, zoom)

for cl in cluster_centroids:
    surreyconmap.addradpoint(cl[0],cl[1],600)

cl_con = np.zeros([len(cluster_centroids),len(cluster_centroids)])
for path_ in pointlst:
    if len(path_)>1:
        for i in range(1,len(path_)):
            if ((path_[i-1][0]-path_[i][0])**2.0+(path_[i-1][1]-path_[i][1])**2.0)**.5 <.15:
                if ((path_[i][0]<49.218415) and (path_[i][1]>-122.937306)) or ((path_[i-1][0]<49.218415) and (path_[i-1][1]>-122.937306)):
                    disti_1 = (path_[i-1][0]-cluster_centroids[:,0])**2.0 + (path_[i-1][1]-cluster_centroids[:,1])**2.0
                    cl_dep = disti_1.argmin()
                    disti = (path_[i][0]-cluster_centroids[:,0])**2.0 + (path_[i][1]-cluster_centroids[:,1])**2.0
                    cl_arr = disti.argmin()
                    cl_con [cl_dep,cl_arr] += 1

for i in range(len(cl_con)):
    for j in range(i,len(cl_con)):
        if i == j:
            surreyconmap.addradpoint(cluster_centroids[i,0],cluster_centroids[i,1],cl_con[i,i]/cl_con.max()*1200)
        elif cl_con[i,j]+cl_con[j,i] > 5:
            surreyconmap.addpath([cluster_centroids[i],cluster_centroids[j]],"#FF0000",cl_con[i,j]+cl_con[j,i])

surreyconmap.draw('./surreyconmap.html')


mymap2 = pygmaps.maps(latitude, longitude, zoom)

for indx in data.index:
    mymap2.addpoint(data.loc[indx]['longitude'],data.loc[indx]['latitude'],"#0000FF")

mymap2.draw('./mymap2.html')



mymap3 = pygmaps.maps(latitude, longitude, zoom)
################################################
#               For Surrey                     #
################################################
#start_lat = 49.005334-.005*6/1.5
#end_lat = 49.385
#lat_int = .005*6
#
#start_lng = -123.293457-.005/2
#end_lng = -122.491221
#lng_int = .007*6
################################################
#               For Downtown                   #
################################################
start_lat = 49.005334
end_lat = 49.385
lat_int = .005/2

start_lng = -123.293457-.005/2
end_lng = -122.491221
lng_int = .007/2

# Grid the map for finding the connection demand for populated areas
mymap3.setgrids(start_lat, end_lat, lat_int, start_lng, end_lng, lng_int)


for indx in data.index:
    mymap3.addradpoint(data.loc[indx]['longitude'],data.loc[indx]['latitude'],95)


mymap3.draw('./mymap3.html')



mymap4 = pygmaps.maps(latitude, longitude, zoom)


# Creating the coordinates of the grids
rlat = np.array([start_lat+float(x1)*lat_int+lat_int/(2.0-x1**0.2/(int((end_lat-start_lat)/lat_int)-1.0)**0.2*0.4) for x1 in range(0, int((end_lat-start_lat)/lat_int))])
rlong = np.array([start_lng+float(x1)*lng_int for x1 in range(0, int((end_lng-start_lng)/lng_int))])+lng_int/2.0

# Creatig routes in each element of the grids
for ilat in range(len(rlat[:-1])):
    for ilng in range(len(rlong[:-1])):
        pnts = data[data['longitude'].between(rlat[ilat],rlat[ilat+1]) & data['latitude'].between(rlong[ilng],rlong[ilng+1])][['latitude','longitude']]
        pnts=pnts.drop_duplicates()
        if len(pnts)>1:
#            y = pnts['longitude'].values
#            arx = np.vstack([pnts['latitude'].values,np.ones(len(pnts))]).T
#            m , c = np.linalg.lstsq(arx,y)[0]

            s,v,d=np.linalg.svd(pnts-np.mean(pnts,axis=0))
            m = d[0,1]/d[0,0]
            c = np.mean(pnts,axis=0)['longitude']-m*np.mean(pnts,axis=0)['latitude']
            
            intnodes = []
                        
            if rlong[ilng]<=((rlat[ilat]-c)/m)<=rlong[ilng+1] : intnodes.append([rlat[ilat],(rlat[ilat]-c)/m])
            if rlong[ilng]<=((rlat[ilat+1]-c)/m)<=rlong[ilng+1] : intnodes.append([rlat[ilat+1],(rlat[ilat+1]-c)/m])
            if rlat[ilat]<=(m*rlong[ilng]+c)<=rlat[ilat+1] : intnodes.append([m*rlong[ilng]+c,rlong[ilng]])
            if rlat[ilat]<=(m*rlong[ilng+1]+c)<=rlat[ilat+1] : intnodes.append([m*rlong[ilng+1]+c,rlong[ilng+1]])
            
            
#            mymap3.addpath(intnodes,"#000000",int(len(pnts)*10/6))
            mymap4.addpath(intnodes,"#000000",int(len(pnts)*4.0/3.0) if len(pnts)<15 else 20)


#mymap4.setgrids(start_lat, end_lat, lat_int, start_lng, end_lng, lng_int)


#for indx in data.index:
#    mymap4.addradpoint(data.loc[indx]['longitude'],data.loc[indx]['latitude'],95)


#mymap3.draw('./mymap3.html')
mymap4.draw('./mymap4.html')


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

data_ = data.drop_duplicates(['latitude','longitude'])

for i in data_.index:
    idx=((data_.loc[i,'latitude']-data3['longitude'])**2.0+(data3['latitude']-data_.loc[i,'longitude'])**2.0).idxmin()
    data3.loc[idx,'repetition'] +=1

outfile = "stopsgtfscrowds.csv"

csvfile = file(outfile, "w")

data3.to_csv(outfile)

csvfile.close()



#############################################################################
##                                                                         ##
##                       connections with user matching                    ##
##                                                                         ##
#############################################################################

mymap5 = pygmaps.maps(latitude, longitude, zoom)

start_lat = 49.005334-.005*6/1.5
end_lat = 49.385
lat_int = .005*3

start_lng = -123.293457-.005/2-.007*3.0/2.0
end_lng = -122.491221
lng_int = .007*3

mymap5.setgrids(start_lat, end_lat, lat_int, start_lng, end_lng, lng_int)

rlat2 = np.array([start_lat+float(x1)*lat_int+lat_int/(2.0) for x1 in range(0, int((end_lat-start_lat)/lat_int))])
rlong2 = np.array([start_lng+float(x1)*lng_int for x1 in range(0, int((end_lng-start_lng)/lng_int))])+lng_int/2.0

#for ii , jj in zip(rlat2,rlong2):
#    mymap5.addpoint(ii,jj,"#0000FF")
    
latnum = len(rlat2)-1
lngnum = len(rlong2)-1

# Connecvtivity matrix is a sparse matrix showing the connection demands between elements of the grid
connectivity_matrix = csr_matrix(np.zeros([latnum*lngnum,latnum*lngnum]))
counter2=0
for path_ in pointlst:
    if len(path_)>0:
        for i in range(1,len(path_)):
            if rlat2[0]<path_[i][0]<rlat2[-1] and rlong2[0]<path_[i][1]<rlong2[-1] and rlat2[0]<path_[i-1][0]<rlat2[-1] and rlong2[0]<path_[i-1][1]<rlong2[-1]:
                connectivity_matrix[lngnum*((path_[i-1][0]-rlat2[0]) // lat_int)+((path_[i-1][1]-rlong2[0]) // lng_int),lngnum*((path_[i][0]-rlat2[0]) // lat_int)+((path_[i][1]-rlong2[0]) // lng_int)] += 1
                counter2 += 1

cm = connectivity_matrix.tocoo()
id_sorted = cm.data.argsort()
val = cm.data[id_sorted]
dep_row = cm.row[id_sorted]
arr_col = cm.col[id_sorted]
counter3=0
for i in range(1,150):
    j_d = dep_row[-i]//lngnum
    i_d = dep_row[-i]%lngnum
    j_a = arr_col[-i]//lngnum
    i_a = arr_col[-i]%lngnum 
    if dep_row[-i] == arr_col[-i]:
        mymap5.addradpoint((rlat2[j_d]+rlat2[j_d+1])/2.0,(rlong2[i_d]+rlong2[i_d+1])/2,val[-i]/val[-1]*200)
    else:
        mymap5.addpath([[(rlat2[j_d]+rlat2[j_d+1])/2.0,(rlong2[i_d]+rlong2[i_d+1])/2.0],[(rlat2[j_a]+rlat2[j_a+1])/2.0,(rlong2[i_a]+rlong2[i_a+1])/2.0]],"#FF0000",int(val[-i]/125.0*30.0))
        counter3+=1
mymap5.draw('./ConnectionsWithUsers.html')

m = cm
heatmap, xedges, yedges = np.histogram2d(m.col, m.row, bins=(latnum,lngnum))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap, extent=extent, origin='lower')
plt.show()











