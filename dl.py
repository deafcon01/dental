from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from dkube import dkubeLoggerHook as logger_hook
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

TRAIN = '../dataset/resized_data'
BATCH_SIZE = 10
IMAGE_SIZE = 224

#Image data generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

#flow images in batch
train_itr = train_generator = train_datagen.flow_from_directory(TRAIN,target_size = (IMAGE_SIZE,IMAGE_SIZE),batch_size =1500,class_mode = 'categorical')

xdata1,ydata1 = train_itr.next()
print(xdata1.shape)
print(ydata1.shape)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
xdata, ydata = sm.fit_sample(xdata1, ydata1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xdata = sc.fit_transform(xdata)

"""


from sklearn.svm import SVC
svc = SVC(kernel='rbf')
from sklearn.metrics import classification_report, confusion_matrix 


  #xtrain = sc.fit(xtrain)
  hist = svc.fit(xtrain,ytrain)
  y_pred = svc.predict(xtest)
  print(confusion_matrix(ytest,y_pred))  
  print(classification_report(ytest,y_pred))
"""

#SET CALLBACKS
#callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor = 0.2, patience = 3, min_lr =0.001),
#        tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience = 3)]

#creating base model
image_shape = (IMAGE_SIZE,IMAGE_SIZE,3)
base_model = tf.keras.applications.vgg19.VGG19(input_shape = image_shape,include_top=False,weights = 'imagenet')

#ADD MORE LAYERS
model = tf.keras.Sequential([
  base_model,
  keras.layers.Dropout(rate = 0.1),
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(4, activation='softmax')
])

fine_tune_at = 3
#print(len(base_model.layers)) #175 for resnet50 #22 for vgg19

base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5,shuffle = True)

#COMPILE
model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=10e-6),metrics=['mae'])

for train,test in kf.split(xdata,ydata):
  model.fit(xdata[train],ydata[train],epochs = 10, batch_size = 10)
  ypred = model.predict_classes(xdata[test])
  print(confusion_matrix(ydata[test],ypred))  
  print(classification_report(ydata[test],ypred)) 





