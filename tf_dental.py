from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
#import resnet

print("tf version:",tf.__version__)

#set Path directories
base_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(base_path,'tooth_train_test_dataset')
train_dir = os.path.join(path,'train_dataset')
validation_dir = os.path.join(path,'val_dataset')
#test_dir = os.path.join(path,'test_dataset/cat1')
#resize images
image_size = 224
batch_size = 16

#Image data generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True,width_shift_range = 0.1,height_shift_range = 0.1,rescale = 1./255)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True,width_shift_range = 0.1,height_shift_range = 0.1,rescale = 1./255)
#test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
#flow images in batch
train_generator = train_datagen.flow_from_directory(train_dir,target_size = (image_size,image_size),batch_size = batch_size,class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size = (image_size,image_size),batch_size = batch_size,class_mode = 'categorical')

#test_generator = test_datagen.flow_from_directory(test_dir,target_size = (image_size,image_size),batch_size = batch_size,class_mode = 'categorical')
#Set callbacks
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor = 0.2, patience = 3, min_lr =0.001),
        tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience = 3)]
#creating base model
image_shape = (image_size,image_size,3)
base_model = tf.keras.applications.vgg19.VGG19(input_shape = image_shape,include_top=False,weights = 'imagenet')
#resnet101 = resnet.ResNet101(input_shape=image_shape,weights='mask_rcnn_coco.h5')
#base_model.summary()

model = tf.keras.Sequential([
  base_model,
  keras.layers.Dropout(rate = 0.1),
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(10, activation='softmax')
])

fine_tune_at = 3
#print(len(base_model.layers)) #175 for resnet50 #22 for vgg19

base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

#compile the model
model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.8,beta_2=0.99,epsilon=10e-6),metrics=["accuracy"])

steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,epochs=12,validation_data=validation_generator,validation_steps=validation_steps,callbacks=callbacks)

model.save_weights('./weights/vgg19_lr_0.0001,epoch_20_dr_0.1_ft_51.h5')
"""
img = image.load_img('./tooth_train_test_dataset/test_dataset/cat1/72.jpg', target_size=(image_size, image_size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

img = image.load_img('./tooth_train_test_dataset/test_dataset/cat1/73.jpg', target_size=(image_size, image_size))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

images = np.vstack([x,y])
classes = model.predict_classes(images, batch_size= 10)
print(classes)
"""
imgs =['./tooth_train_test_dataset/test_dataset/cat1/72.jpg',
'./tooth_train_test_dataset/test_dataset/cat1/73.jpg',
'./tooth_train_test_dataset/test_dataset/cat2/219.jpg',
'./tooth_train_test_dataset/test_dataset/cat2/220.jpg',
'./tooth_train_test_dataset/test_dataset/cat3/44.jpg',
'./tooth_train_test_dataset/test_dataset/cat3/45.jpg',
'./tooth_train_test_dataset/test_dataset/cat4/139.jpg',
'./tooth_train_test_dataset/test_dataset/cat4/140.jpg']

for img in imgs:
  x = image.load_img(img,target_size=(image_size,image_size))
  x = image.img_to_array(x)
  x = np.expand_dims(x,axis=0)
  pred_class = model.predict_classes(x, batch_size=10)
  print(pred_class) 
"""
def get_session(gpu_fraction=0.3):
  num_threads = os.environ.get('OMP_NUM_THREADS')
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

  if num_threads:
    return tf.Session(config=tf.ConfigProto(
      gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
  else:
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

  config = tf.ConfigProto( device_count = {'GPU': 1 } )
  sess = tf.Session(config=config)
  KTF.set_session(get_session())

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()
"""
