import numpy as np
from glob import glob
#from pandas import DataFrame
import csv
import h5py
import cv2
import os
import matplotlib.pylab as plt
#%%

path2annotations='/media/mra/win71/data/misc/kaggle/intel/data/annotations/'

# path to train data
path2train_hdf5='/media/mra/win71/data/misc/kaggle/intel/data/train.hdf5'

#%%

# get list of tsv files
tsv_list=glob(path2annotations+'*.tsv')


# read file
k=0
lines=[]
with open(tsv_list[0]) as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        k+=1
        line0=line[0]
        
        lines.append(line)
        
        
#%%        
bb={}        
path2annotations_hdf5='/media/mra/win71/data/misc/kaggle/intel/data/train_annotations.hdf5'

if not os.path.exists(path2annotations_hdf5):
    ff_w=h5py.File(path2annotations_hdf5,'w')
    
    # train data hdf5 
    ff_train=h5py.File(path2train_hdf5,'r')
    
    for tsv_file in tsv_list:
        with open(tsv_file) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                row0=row[0].split("\\")
                type_name=row0[0]
                row1=row0[1]
                row1=row1.split(" ")
                file_name=row1[0]
                nb_rec=int(row1[1])
                for k in range(nb_rec):
                    x,y,w,h=row1[4*k+2:4*k+6]
                    x=int(x)
                    y=int(y)
                    w=int(w)
                    h=int(h)
                    #print type_name,file_name,x,y,w,h
                    bb[file_name]=(x,y,w,h)
                
                # create a group dataset to store bounding box dimensions and image size
                grp=ff_w.create_group(file_name)        
                grp['bb']=(x,y,w,h)                
                grp['Xshape']=ff_train[file_name]['X'].shape
                #ff_w[file_name]=(x,y,w,h)                
    ff_w.close()           
else: 
    print '%s exsits!!' %path2annotations_hdf5 
    ff_r=h5py.File(path2annotations_hdf5,'r')            
    print 'number of keys:', len(ff_r.keys())
#%%

# stats
minX=[]
minY=[]
maxX=[]
maxY=[]
for key in ff_r.keys():
    print key,ff_r[key]['Xshape'].value,ff_r[key]['bb'].value
    x,y,w,h=ff_r[key]['bb'].value
    minX.append(x)
    minY.append(y)
    maxX.append(x+w)
    maxY.append(y+h)
    
    
plt.subplot(2,2,1)
plt.hist(minX)
plt.title('minX')

plt.subplot(2,2,2)
plt.hist(minY)
plt.title('minY')

plt.subplot(2,2,3)
plt.hist(maxX)
plt.title('maxX')

plt.subplot(2,2,4)
plt.hist(maxY)
plt.title('maxY')
