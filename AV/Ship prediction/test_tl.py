# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:05:19 2019

@author: Gireesh Sundaram
"""

import pandas as pd

from keras.layers import Input, TimeDistributed, Bidirectional, Conv2D, BatchNormalization, MaxPooling2D, Flatten, LSTM, Dense, Lambda, GRU, Activation, Dropout
from keras import applications
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.merge import add, concatenate

from config import height, width

import numpy as np
import cv2

#%%
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (width, height, 3))

for layer in model.layers[:14]:
    layer.trainable = False

model.summary()

x = model.output
x = Flatten()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(32, activation="relu")(x)
dense = Dense(5, name = "dense")(x)
y_pred = Activation('softmax', name='label')(dense)

model_final = Model(input = model.input, output = y_pred)
model_final.summary()


#%%
filepath="gdrive/My Drive/Colab/Ship/Checkpoints_tl/weights.0.997a-0.019l.hdf5"

model_final.load_weights(filepath)

#%%
def read_one_image(filename):
    image = cv2.imread(filename)
    h, w, f = np.shape(image)
    
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    h, w, f = np.shape(image)
    
    return image

#%%
test = pd.read_csv("gdrive/My Drive/Colab/Ship/Data/test_ApKoW4T.csv")

image_out = np.zeros((len(test), height, width, 3))

for i in range(0, len(test)):
    path = "gdrive/My Drive/Colab/Ship/Data/Image/" + test["image"][i]
    print(path)
    image_out[i] = read_one_image(path)


#%%
arr = model_final.predict(image_out)
predx = (np.argmax(arr, 1) + 1)

#%%
test["category"] = predx

test.to_csv("gdrive/My Drive/Colab/Ship/Data/Submission1.csv", index = False)
