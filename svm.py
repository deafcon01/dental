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
DATUMS_PATH = os.getenv('DATUMS_PATH', None)
DATASET_NAME = os.getenv('DATASET_NAME', None)
TF_TRAIN_STEPS = int(os.getenv('TF_TRAIN_STEPS',1000))
MODEL_DIR = os.getenv('OUT_DIR', None)
DATA_DIR = "{}/{}/{}".format(DATUMS_PATH, DATASET_NAME,'small_data')
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
#Image data generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True,width_shift_range = 0.1,height_shift_range = 0.1,rescale = 1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True,width_shift_range = 0.1,height_shift_range = 0.1,rescale = 1./255)
#flow images in batch
train_itr = train_generator = train_datagen.flow_from_directory(TRAIN,target_size = (IMAGE_SIZE,IMAGE_SIZE),batch_size =400,class_mode = 'categorical')

val_itr = validation_generator = validation_datagen.flow_from_directory(VAL,target_size = (IMAGE_SIZE,IMAGE_SIZE),batch_size = 86,class_mode = 'categorical')

#NUMPY ARRAY
xtrain,ytrain = train_itr.next()
xval,yval = val_itr.next()

print(xtrain.shape)
print(ytrain.shape)
print(xval.shape)
print(yval.shape)


#PREPROCESSING
nsamples , nx ,ny ,nz =xtrain.shape
img_xtrain = xtrain.reshape((nsamples,nx*ny*nz))

nsamples , nx ,ny ,nz =xval.shape
img_xval = xval.reshape((nsamples,nx*ny*nz))

img_ytrain = np.empty(shape = (400,))
for i in range(0,400):
  img_ytrain[i] = np.argmax(ytrain[i])

img_yval = np.empty(shape = (86,))
for i in range(0,86):
  img_yval[i] = np.argmax(yval[i])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
img_xtrain = sc.fit_transform(img_xtrain)
img_xval = sc.transform(img_xval)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
hist = svc.fit(img_xtrain,img_ytrain)

y_pred = svc.predict(img_xval)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(img_yval,y_pred))  
print(classification_report(img_yval,y_pred))


