from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from keras import applications
import operator
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
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
from keras.utils import np_utils
import numpy as np
import cv2
import keras
import json
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from keras import applications
import operator
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import os

target_size = 224,224
img_width, img_height = 224, 224
num_classes=10
img_path = 'img_100670.jpg'
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
#train_data_dir = '/storage/work/pug64/trial2vgg/imgs/train/' # the path to the training data
#validation_data_dir = '/storage/work/pug64/trial2vgg/imgs/validation/' # the path to the training data
#test_data_dir = '/storage/work/pug64/trial2vgg/imgs/test/' # the path to the test data

def predict_class(img_path):
    
    # prepare image for classification using keras utility functions
    image = load_img(img_path, target_size=target_size)
    image_arr = img_to_array(image) # convert from PIL Image to NumPy array
    image_arr /= 255
    image_arr = np.expand_dims(image_arr, axis=0)
    # print(image.shape)
    predicted = model.predict(image_arr)
    # predicted_onehot = to_categorical(predicted, num_classes=num_classes)
    return np.asarray(predicted[0]) # float32

model = load_model('model_resnet2.h5')
predicted = predict_class(img_path)
predicted = np.asarray(['%.1f'%num for num in predicted]).astype('str')
print('True Class : Safe_driving')
print('Predicted Class : ' + class_labels[predicted.argmax(0)] + ' with probability ' + max(predicted))
