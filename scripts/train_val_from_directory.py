#%% classify positive nodes and negative nodes
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
#from skimage import measure
import models
import utils
#from keras import backend as K
from keras.utils import np_utils
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import h5py
from glob import glob
#from image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
#from keras.callbacks import ModelCheckpoint
import datetime
import pandas as pd
#%%
# settings

# fold number
foldnm=1

# pre-processed data dimesnsion
z,h,w=3,256,256


train_data_dir = '/media/mra/My Passport/Kaggle/intel/data/train'
validation_data_dir = '/media/mra/My Passport/Kaggle/intel/data/train'
nb_train_samples = 1480
nb_validation_samples = 1480


# batch size
batch_size=8

# classes 
classes=['Type_1','Type_2','Type_3']

# number of classes
num_classes=3

# seed point
seed = 2017
seed = np.random.randint(seed)

# exeriment name to record weights and scores
experiment='fold'+str(foldnm)+'_classify_from_directory'+'_hw_'+str(h)+'by'+str(w)
print ('experiment:', experiment)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# pre train
pre_train=False
#%%

########## log
path2logs='./output/logs/'
now = datetime.datetime.now()
info='log_trainval_from_dir_'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)


#%%

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        rescale=1/255.,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=0.05,
        channel_shift_range=0.0,
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        dim_ordering='th') 

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(w, h),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(w, h),
    batch_size=batch_size,
    class_mode='categorical')

#%%
print('-'*30)

# training params
params_train={
    'h': h,
    'w': w,
    'z': z,
    'c':1,           
    'learning_rate': 5e-5,
    'optimizer': 'Adam',
    'loss': 'categorical_crossentropy',
    'nbepoch': 2000,
    'num_classes': num_classes,
    'nb_filters': 32,    
    'max_patience': 30,
    'stride': 2,    
        }

#model = models.model(params_train)
model = models.vgg_model(params_train)
model.summary()

# path to weights
path2weights=weightfolder+"/weights.hdf5"

#%%

print ('training in progress ...')

# checkpoint settings
#checkpoint = ModelCheckpoint(path2weights, monitor='val_loss', verbose=0, save_best_only='True',mode='min')

# load last weights
if pre_train:
    if  os.path.exists(path2weights) and pre_train:
        model.load_weights(path2weights)
        print 'previous weights loaded!'
    else:
        raise IOError('weights does not exist!')

# path to csv file to save scores
path2scorescsv = weightfolder+'/scores.csv'
first_row = 'train,test'
with open(path2scorescsv, 'w+') as f:
    f.write(first_row + '\n')
    
    
# Fit the model
start_time=time.time()
scores_test=[]
scores_train=[]
best_score = 1e6
previous_score = 1e6
patience = 0

#batches =0
#for X_batch, y_batch in train_generator:
#    print X_batch.shape,batches
#    batches += 1
#    if batches >= (nb_train_samples) / batch_size:
#        # we need to break the loop by hand because
#        # the generator loops indefinitely
#        break

for epoch in range(params_train['nbepoch']):

    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)

    hist=model.fit_generator(
        train_generator,
        samples_per_epoch=185*batch_size,
        nb_epoch=1,verbose=0)
    
    
    # evaluate on test and train data
    score_test=model.evaluate_generator(validation_generator,val_samples=nb_validation_samples // batch_size)
    
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

print ('best scores train: %.5f' %(np.min(scores_train)))
print ('best scores test: %.5f' %(np.min(scores_test)))          
          
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


score_test=model.evaluate(X_test,y_test,verbose=0,batch_size=8)
#score_train=model.evaluate(X_batch,y_batch,verbose=0,batch_size=8)
#print ('score_train: %.2f, score_test: %.2f' %(score_train,score_test))
print ('score_test: %.2f' %(score_test))

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

# normalize
X_test_leader=preprocess_input(X_test_leader)

# prediction
y_pred=model.predict(X_test_leader)
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



