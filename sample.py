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
tf.logging.set_verbosity(tf.logging.INFO)

DATUMS_PATH = os.getenv('DATUMS_PATH', None)
DATASET_NAME = os.getenv('DATASET_NAME', None)
TF_TRAIN_STEPS = int(os.getenv('TF_TRAIN_STEPS',1000))
MODEL_DIR = os.getenv('OUT_DIR', None)
DATA_DIR = "{}/{}/{}".format(DATUMS_PATH, DATASET_NAME,'small_data')
BATCH_SIZE = int(os.getenv('TF_BATCH_SIZE', 16))
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
train_generator = train_datagen.flow_from_directory(TRAIN,target_size = (IMAGE_SIZE,IMAGE_SIZE),batch_size = BATCH_SIZE,class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_directory(VAL,target_size = (IMAGE_SIZE,IMAGE_SIZE),batch_size = BATCH_SIZE,class_mode = 'categorical')

#SET CALLBACKS
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor = 0.2, patience = 3, min_lr =0.001),
        tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience = 3)]

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

#COMPILE
model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.8,beta_2=0.99,epsilon=10e-6),metrics=["accuracy"])

steps_per_epoch = train_generator.n // BATCH_SIZE
validation_steps = validation_generator.n // BATCH_SIZE

history = model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,epochs=EPOCHS,validation_data=validation_generator,validation_steps=validation_steps,callbacks=callbacks)
logging_hook = logger_hook({"history":history})

print(history)
