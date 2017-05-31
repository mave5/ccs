import numpy as np
#import csv
import h5py
import os
#import cv2
from glob import glob
import matplotlib.pylab as plt
from PIL import Image
#import pandas as pd
import ntpath
#from skimage.draw import circle

#%%
H,W=512,512

# path to data
path2train='/media/mra/My Passport/Kaggle/intel/data/train/'
path2train_extra='/media/mra/My Passport/Kaggle/intel/data/extra/'
path2test='/media/mra/My Passport/Kaggle/intel/data/test/'

path2hdf5='/media/mra/win71/data/misc/kaggle/intel/data/train.hdf5'

path2testhdf5='/media/mra/win71/data/misc/kaggle/intel/data/test.hdf5'
path2trainextra_hdf5='/media/mra/win71/data/misc/kaggle/intel/data/train_extra.hdf5'
    
#%% collect train data

# path to types
path2types=glob(path2train+'Type*')

if not os.path.exists(path2hdf5):

    # create a hdf5 file
    ff_w=h5py.File(path2hdf5,'w')
    
    # list of subsets
    for p2type in path2types:
        type_nm=ntpath.basename(p2type)    
        subset_list=glob(p2type+'/*.jpg')
        subset_list.sort()
        print 'total subsets: %s' %len(subset_list)
    
        for k,path2img in enumerate(subset_list):
            print k,path2img
            img_id=ntpath.basename(path2img)
            img=Image.open(path2img)
            #plt.imshow(img)
            
            # create group
            try:
                grp=ff_w.create_group(img_id)
                grp['X']=img
                grp['y']=type_nm
            except: 
                print 'skip this!'
    
    ff_w.close()
else:
    print 'train hdf5 exist!!!!!!'

## verify
ff_r=h5py.File(path2hdf5,'r')
print ff_r.keys()
keys=ff_r.keys()
X=ff_r[keys[0]]['X']
y=ff_r[keys[0]]['y']
plt.imshow(X)
ff_r.close()
#%%

# collect test data

if not os.path.exists(path2testhdf5):

    # create a hdf5 file
    ff_w=h5py.File(path2testhdf5,'w')
    
    test_list=glob(path2test+'/*.jpg')
    test_list.sort()
    print 'total subsets: %s' %len(test_list)
    
    
    
    for k,path2img in enumerate(test_list):
        print k,path2img
        img_id=ntpath.basename(path2img)
        img=Image.open(path2img)
        #plt.imshow(img)
        
        # create group
        try:
            grp=ff_w.create_group(img_id)
            grp['X']=img
            #grp['y']=type_nm
        except: 
            print 'skip this!'
    
    ff_w.close()
else:
    print 'test hdf5 exists!!!!!!'
#%%

# path to types
path2types=glob(path2train_extra+'Type*')

if not os.path.exists(path2trainextra_hdf5):

    # create a hdf5 file
    ff_w=h5py.File(path2trainextra_hdf5,'w')
    
    # list of subsets
    for p2type in path2types:
        type_nm=ntpath.basename(p2type)    
        subset_list=glob(p2type+'/*.jpg')
        subset_list.sort()
        print 'total subsets: %s' %len(subset_list)
    
        for k,path2img in enumerate(subset_list):
            print k,path2img
            img_id=ntpath.basename(path2img)
            try:
                img=Image.open(path2img)
            except:
                print 'skip this!'
                continue
            #plt.imshow(img)
            
            # create group
            try:
                grp=ff_w.create_group(img_id)
                grp['X']=img
                grp['y']=type_nm
            except: 
                print 'skip this!'
    
    ff_w.close()
else:
    print 'train hdf5 exist!!!!!!'

## verify
ff_r=h5py.File(path2trainextra_hdf5,'r')
print ff_r.keys()
keys=ff_r.keys()
X=ff_r[keys[0]]['X']
y=ff_r[keys[0]]['y']
plt.imshow(X)
ff_r.close()


#%%
ff_r=h5py.File(path2testhdf5,'r')
print ff_r.keys()
keys=ff_r.keys()
X=ff_r[keys[0]]['X']
#y=ff_r[keys[0]]['y']
plt.imshow(X)
ff_r.close()

