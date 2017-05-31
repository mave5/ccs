#%% classify positive nodes and negative nodes
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
#from skimage import measure
#import models
#from inception_v4 import create_inception_v4
import utils
#from keras import backend as K
from keras.utils import np_utils
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import h5py
from glob import glob
from image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import datetime
import pandas as pd

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
#%%
# settings

# fold number
foldnm=1

# path to data
path2trainhdf5='/media/mra/win71/data/misc/kaggle/intel/data/train.hdf5'
path2testhdf5='/media/mra/win71/data/misc/kaggle/intel/data/test.hdf5'

# path train test
path2root='/media/mra/win71/data/misc/kaggle/intel/data/'

#%%

# pre-processed data dimesnsion
z,h,w=3,299,299

# batch size
bs=8

# number of classes
num_classes=3

# seed point
seed = 2017
seed = np.random.randint(seed)

# exeriment name to record weights and scores
experiment='fold'+str(foldnm)+'_inceptionv3'+'_hw_'+str(h)+'by'+str(w)
print ('experiment:', experiment)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# data augmentation 
augmentation=True

# pre train
pre_train=False
#%%

########## log
path2logs='./output/logs/'
now = datetime.datetime.now()
info='log_trainval_'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)


#%%

# random data generator
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=0.01,
        channel_shift_range=0.0,
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        dim_ordering='th') 


def iterate_minibatches(inputs1 , targets,  batchsize, shuffle=True, augment=True):
    assert len(inputs1) == len(targets)
    if augment==True:
        if shuffle:
            indices = np.arange(len(inputs1))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            x = inputs1[excerpt]
            y = targets[excerpt] 
            for  xxt,yyt in datagen.flow(x, y , batch_size=x.shape[0]):
                x = xxt.astype(np.float32) 
                y = yyt 
                break
    else:
        x=inputs1
        y=targets

    #yield x, np.array(y, dtype=np.uint8)         
    return x, np.array(y, dtype=np.uint8)         


# load train data
def load_data(ids,data_type='train'):
    X=[]
    y=[]
    
    if data_type=='train':
        ff=h5py.File(path2trainhdf5,'r')
        #print len(ff_test.keys())
    else:
        ff=h5py.File(path2testhdf5,'r')        
    
    for k,id1 in enumerate(ids):
        print k,id1
        try:
            X0=ff[id1]['X'].value
            X0 = cv2.resize(X0, (w, h), interpolation=cv2.INTER_CUBIC)
            #print X0.shape
            if data_type=='train':
                y0=ff[id1]['y'].value
            else:
                y0=0
            X.append(X0)
            y.append(y0)
        except:
            print 'skip this!'
    ff.close()    
    X=np.stack(X) 
    X=np.transpose(X,(0,3,1,2))
    y=np.hstack(y)
    
    # convert labels to numbers
    y[y=='Type_1']=0
    y[y=='Type_2']=1
    y[y=='Type_3']=2
    y=y.astype('int')
    
    return X,y

#%%

path2traintest=path2root+'traintest'+str(h)+'by'+str(w)+'.hdf5'
if not os.path.exists(path2traintest):

    print('-'*30)
    ff_train=h5py.File(path2trainhdf5,'r')
    ids=ff_train.keys()
    print 'total train', len(ff_train.keys())
    
    # train test split
    ids_train, ids_test, _, _ = train_test_split(ids, ids, test_size=0.1, random_state=42)

    X_train,y_train=load_data(ids_train)
    ff_traintest=h5py.File(path2traintest,'a')
    ff_traintest['X_train']=X_train
    ff_traintest['y_train']=y_train    
    
    X_test,y_test=load_data(ids_test)
    ff_traintest['X_test']=X_test
    ff_traintest['y_test']=y_test    
    ff_traintest.close()
else:
    ff_traintest=h5py.File(path2traintest,'r')
    X_train=ff_traintest['X_train']
    y_train=ff_traintest['y_train']    
    
    X_test=ff_traintest['X_test']
    y_test=ff_traintest['y_test']    


# normalization
X_train_cp=np.array(X_train,'float32')
X_train_cp = preprocess_input(X_train_cp)

X_test_cp=np.array(X_test,'float32')
X_test_cp = preprocess_input(X_test_cp)

    #%%
print('-'*30)

# training params
params_train={
    'h': h,
    'w': w,
    'z': z,
    'c':1,           
    'learning_rate': 3e-5,
    'optimizer': 'Adam',
    'loss': 'categorical_crossentropy',
    'nbepoch': 2000,
    'nb_classes': num_classes,
    'nb_filters': 32,    
    'max_patience': 20    
        }


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
#x = Dense(512, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)


# this is the model we will train
model = Model(input=base_model.input, output=predictions)
#odel = create_inception_v4(params_train)
#model = models.model(params_train)
model.summary()


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# path to weights
path2weights=weightfolder+"/weights.hdf5"

#%%

print ('training in progress ...')

# checkpoint settings
#checkpoint = ModelCheckpoint(path2weights, monitor='val_loss', verbose=0, save_best_only='True',mode='min')

# load last weights
if pre_train:
    if  os.path.exists(path2weights): 
        model.load_weights(path2weights)
        print 'weights loaded!'
    else:
        raise IOError

# path to csv file to save scores
path2scorescsv = weightfolder+'/scores.csv'
first_row = 'train,test'
with open(path2scorescsv, 'w+') as f:
    f.write(first_row + '\n')
    
    
# Fit the model
start_time=time.time()
scores_test=[]
scores_train=[]
if params_train['loss']=='dice': 
    best_score = 0
    previous_score = 0
else:
    best_score = 1e6
    previous_score = 1e6
patience = 0

# find nonzero diameters
pos_inds=np.nonzero(y_train)[0]
neg_inds=np.where(y_train==0)[0]

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


for epoch in range(params_train['nbepoch']):

    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)


    # data augmentation
    #bs2=256
    #for k in range(0,len(X_train),bs2):
        #X_batch=X_train_cp[k:k+bs2]#[:,np.newaxis]
        #y_batch=y_train_cp[k:k+bs2]
        
    # augmentation
    X_batch,_=iterate_minibatches(X_train_cp,X_train_cp,X_train_cp.shape[0],shuffle=False,augment=True)  
        
    hist=model.fit(X_batch, y_train, nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)
        #print 'partial loss:', hist.history['loss']
    
    # evaluate on test and train data
    #score_test=[]
    #for k2 in range(0,X_test.shape[0],bs):
    score_test=model.evaluate(X_test_cp,y_test,verbose=0,batch_size=bs)
    #score_test.append(tmp)
    #score_test=np.mean(np.array(score_test),axis=0)


    #if params_train['loss']=='dice': 
        #score_test=score_test[1]   
    score_train=hist.history['loss']
   
    print ('score_train: %s, score_test: %s' %(score_train,score_test))
    scores_test=np.append(scores_test,score_test)
    scores_train=np.append(scores_train,score_train)    

    # check if there is improvement
    if params_train['loss']=='dice': 
        if (score_test>=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)       
            model.save(weightfolder+'/model.h5')
            
        # learning rate schedule
        if score_test<=previous_score:
            #print "Incrementing Patience."
            patience += 1
    else:
        if (score_test<=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)  
            model.save(weightfolder+'/model.h5')
            
        # learning rate schedule
        if score_test>previous_score:
            #print "Incrementing Patience."
            patience += 1
    # save anyway    
    #model.save_weights(path2weights)      
            
    if patience == params_train['max_patience']:
        params_train['learning_rate'] = params_train['learning_rate']/2
        print ("Upating Current Learning Rate to: ", params_train['learning_rate'])
        model.optimizer.lr.set_value(params_train['learning_rate'])
        print ("Loading the best weights again. best_score: ",best_score)
        model.load_weights(path2weights)
        patience = 0
    
    # save current test score
    previous_score = score_test    
    
    # real time plot
    #plt.plot([e],[score_train],'b.')
    #plt.plot([e],[score_test],'b.')
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    #sys.stdout.flush()
    
    # store scores into csv file
    with open(path2scorescsv, 'a') as f:
        string = str([score_train,score_test])
        f.write(string + '\n')
       

print ('model was trained!')
elapsed_time=(time.time()-start_time)/60
print ('elapsed time: %d  mins' %elapsed_time)          
#%%

plt.figure(figsize=(15,10))
plt.plot(scores_test)
plt.plot(scores_train)
plt.title('train-validation progress',fontsize=20)
plt.legend(('test','train'), loc = 'lower right',fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.ylabel('loss',fontsize=20)
plt.grid(True)
plt.show()
plt.savefig(weightfolder+'/train_val_progress.png')

print ('best scores train: %.5f' %(np.max(scores_train)))
print ('best scores test: %.5f' %(np.max(scores_test)))          
          
#%%
# loading best weights from training session
print('-'*30)
print('Loading saved weights...')
print('-'*30)
# load best weights

if  os.path.exists(path2weights):
    model.load_weights(path2weights)
    print 'weights loaded!'
else:
    raise IOError


score_test=model.evaluate(np.array(X_test)/255.,y_test,verbose=0,batch_size=8)
score_train=model.evaluate(X_batch/255.,y_batch,verbose=0,batch_size=8)
print ('score_train: %.2f, score_test: %.2f' %(score_train,score_test))

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
#y_pred=model.predict(preprocess(X_test,Y_test,param_prep)[0])
#y_pred=model.predict(preprocess(X_train,Y_train,param_prep)[0])
#%%

# test on leaderboard data
df = pd.read_csv('../data/sample_submission.csv')
print('Number of training patients: {}'.format(len(df)))
df.head()

# load data
X_test_leader,_=load_data(df.image_name,'test')

# prediction
y_pred=model.predict(X_test_leader/255.)
print y_pred.shape
df.Type_1=y_pred[:,0]
df.Type_2=y_pred[:,1]
df.Type_3=y_pred[:,2]

# make submission
now = datetime.datetime.now()
info='cnn_'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('./output/submission', 'submission_' + suffix + '.csv')

df.to_csv(sub_file, index=False)
print(df.head()) 
#%%


