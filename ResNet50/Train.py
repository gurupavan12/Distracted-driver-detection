import os
import h5py
import numpy as np
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
plt.switch_backend('agg')
import math
import json
import h5py
import keras
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU
from keras.layers.core import Dense, Dropout, Reshape
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
import keras.backend as k
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, regularizers
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
#from keras.utils.visualize_util import model_to_dot
import numpy as np
import cv2
import keras
import json
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model


img_width, img_height = 224, 224
num_classes=10

train_data_dir = '/storage/work/pug64/trial2vgg/imgs/train/' # the path to the training data
validation_data_dir = '/storage/work/pug64/trial2vgg/imgs/validation/' # the path to the training data
test_data_dir = '/storage/work/pug64/trial2vgg/imgs/test/' # the path to the test data

nb_train_samples = 15688  #Using count_num_imgs.py
nb_validation_samples = 2246
nb_test_samples = 4490

# number of epoches for training
nb_epoch = 5
filepath = '/storage/work/pug64/trial2vgg/model_resnet_checkpoints-{epoch:02d}-{loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')

# build the ResNet50 model
model_res = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=model_res.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

for layer in model_res.layers[:-5]:
    layer.trainable=False

# connect the two models 
model = Sequential()
model.add(model_res)
model.add(top_model)
model.summary()

'''Uncomment the below line if you want to load the model directly and
 comment all the lines on top until  Line 65'''

#model = load_model('model_resnet.h5')  

# compile the model 
model.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# augmentation configuration for training data
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, rotation_range = 20)
# augmentation configuration for validation data 
#val_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.1, zoom_range=0.1, rotation_range = 15)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# training data generator from folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), 
                                                  batch_size=32, class_mode='categorical')

# validation data generator from folder
validation_generator = train_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), 
                                                       batch_size=32, class_mode='categorical')

# fit the model
history = model.fit_generator(train_generator, samples_per_epoch=nb_train_samples,
                              nb_epoch=nb_epoch,validation_data=validation_generator,
                              nb_val_samples=nb_validation_samples/32)


class LifecycleCallback(keras.callbacks.Callback):
    
    
    def on_epoch_begin(self, epoch, logs = {}):
        pass
    def on_epoch_end(self, epoch, logs = {}):
        global threshold
        threshold = 1 / (epoch  + 1)

    def on_batch_begin(self, batch, logs = {}):
        pass
    
    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))
        
    def on_train_begin(self, logs = {}):
        print('BEGIN TRAINING!...')
        self.losses = []
        
    def on_train_end(self, logs = {}):
        print('END TRAINING')

# constructor for callbacks
lifecycle_callback = LifecycleCallback()

model.save('/storage/work/pug64/trial2vgg/model_resnet2.h5')
test_generator = test_datagen.flow_from_directory(test_data_dir, 
                                                  target_size=(img_height, img_width),
                                                  batch_size=32, class_mode='categorical')

test_labels = test_generator.classes # actual class number
test_labels_onehot = to_categorical(test_labels, num_classes=num_classes)

print('saved model to disk')
#print(history.history.keys())

model_json = model.to_json()
with open('/storage/work/pug64/trial2vgg/model_resnet2.json', "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights('/storage/work/pug64/trial2vgg/model_resnet2_weights.h5')
print('saved model to disk')

# Plots
#plt.figure()
#print(history.history.keys())

# summarize history for epoch loss
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#
#plt.savefig('/storage/work/pug64/trial2vgg/model_resnet_loss_eps.png')





# summarize history for accuracy
# summarize history for epoch loss
#plt.plot(history.history['acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')

#plt.savefig('/storage/work/pug64/trial2vgg/model_resnet_acc_eps_.png')

