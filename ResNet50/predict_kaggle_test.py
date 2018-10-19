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
import argparse

a = argparse.ArgumentParser(description="Generate predictions for test images provided by Kaggle.")
a.add_argument("-f", "--filename", help='csv file name to save the generated predictions to (default: predictions.csv)', default='predictions_resnet.csv')
args = a.parse_args()

filename = args.filename

data_path = '/storage/work/pug64/jia_dong/input/test'
csv_header = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
all_entries = []
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']

target_size= 224, 224

model = load_model('model_resnet2.h5')
model.summary()
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

for i, file in enumerate(os.listdir(data_path)):
    file_name = os.fsdecode(file)

    if file_name.endswith(".jpg"):
        
        print(i, " ", file_name, "...")
        img_path = (os.path.join(data_path, file_name))
        
        predicted = predict_class(img_path)
        predicted = np.asarray(['%.1f'%num for num in predicted]).astype('str')
        
        entry = np.concatenate((np.array([file_name]), predicted))
        all_entries.append(entry)
        
all_entries = (np.asarray(all_entries))
all_entries = all_entries[np.argsort(all_entries[:,0])]
import csv
with open(filename, 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(csv_header)
    for row in all_entries:
        filewriter.writerow(row)