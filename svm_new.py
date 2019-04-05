#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dkube import dkubeLoggerHook as logger_hook
import argparse
import os
import sys

import tensorflow as tf
from tensorflow import keras
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
#import dataset
#tf.logging.set_verbosity(tf.logging.INFO)


# In[2]:


DATUMS_PATH = os.getenv('DATUMS_PATH', None)
DATASET_NAME = os.getenv('DATASET_NAME', None)
TF_TRAIN_STEPS = int(os.getenv('TF_TRAIN_STEPS',1000))
MODEL_DIR = os.getenv('OUT_DIR', None)
DATA_DIR = "{}/{}/{}".format(DATUMS_PATH, DATASET_NAME,'small_train')
BATCH_SIZE = 5
EPOCHS = int(os.getenv('TF_EPOCHS', 10))
TF_MODEL_DIR = MODEL_DIR
steps_epoch = 0
summary_interval = 100
print ("ENV, EXPORT_DIR:{}, DATA_DIR:{}".format(MODEL_DIR, DATA_DIR))
print ("TF_CONFIG: {}".format(os.getenv("TF_CONFIG", '{}')))
TRAIN = "{}/{}".format(DATA_DIR,"train")
VAL = "{}/{}".format(DATA_DIR,"val")
TEST = "{}/{}".format(DATA_DIR,"test")
IMAGE_SIZE = 224


# In[3]:


#Image data generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True,width_shift_range = 0.2,height_shift_range = 0.2,rescale = 1./255,zoom_range = 0.2, fill_mode ='nearest')

validation_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True,width_shift_range = 0.2,height_shift_range = 0.2,rescale = 1./255,zoom_range = 0.2, fill_mode = 'nearest')


# In[4]:


#flow images in batch
train_itr = train_generator = train_datagen.flow_from_directory(TRAIN,target_size = (IMAGE_SIZE,IMAGE_SIZE),batch_size =400,class_mode = 'categorical')

#val_itr = validation_generator = validation_datagen.flow_from_directory(VAL,target_size = (IMAGE_SIZE,IMAGE_SIZE),batch_size = 86,class_mode = 'categorical')


# In[5]:


#NUMPY ARRAY
xtrain,ytrain = train_itr.next()
print(xtrain.shape)
print(ytrain.shape)


# In[6]:


#PREPROCESSING
nsamples , nx ,ny ,nz =xtrain.shape
img_xtrain = xtrain.reshape((nsamples,nx*ny*nz))


# In[7]:


img_ytrain = np.empty(shape = (400,))
for i in range(0,400):
  img_ytrain[i] = np.argmax(ytrain[i])


# In[8]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
img_xtrain = sc.fit_transform(img_xtrain)


# In[9]:


from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5,shuffle = True)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
from sklearn.metrics import classification_report, confusion_matrix 

for train,test in kf.split(img_xtrain,img_ytrain):
  xtrain, xtest = img_xtrain[train],img_xtrain[test]
  ytrain, ytest = img_ytrain[train], img_ytrain[test]
  #xtrain = sc.fit(xtrain)
  hist = svc.fit(xtrain,ytrain)
  y_pred = svc.predict(xtest)
  print(confusion_matrix(ytest,y_pred))  
  print(classification_report(ytest,y_pred))


# In[10]:


import sklearn
print(sklearn.__version__)


# In[11]:


from sklearn.model_selection import GridSearchCV
c = [0.0001,0.001]
gamma =  [0.001, 0.01]
param_grid = {'C': c, 'gamma' : gamma}
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(img_xtrain,img_ytrain)
grid_search.best_params_


# In[ ]:




