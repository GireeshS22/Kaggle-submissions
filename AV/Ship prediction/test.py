# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:05:19 2019

@author: Gireesh Sundaram
"""

import pandas as pd

from keras.layers import Input, TimeDistributed, Bidirectional, Conv2D, BatchNormalization, MaxPooling2D, Flatten, LSTM, Dense, Lambda, GRU, Activation
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.merge import add, concatenate

from config import height, width

import numpy as np
import cv2

#%%
input_data = Input(shape= (height, width, 3), name= "the_input")

conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv1")(input_data)
bn = BatchNormalization(name = "batch_norm1")(conv)
conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv2")(bn)
bn = BatchNormalization(name = "batch_norm2")(conv)
pooling = MaxPooling2D(pool_size=(2,2), name = "max_pool1")(bn)

conv = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv3")(pooling)
bn = BatchNormalization(name = "batch_norm3")(conv)
conv = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv4")(bn)
bn = BatchNormalization(name = "batch_norm4")(conv)
pooling = MaxPooling2D(pool_size=(2,2), name = "max_pool2")(bn)

conv = Conv2D(filters=256, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv5")(pooling)
bn = BatchNormalization(name = "batch_norm5")(conv)
conv = Conv2D(filters=256, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv6")(bn)
bn = BatchNormalization(name = "batch_norm6")(conv)
conv = Conv2D(filters=256, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv7")(bn)
bn = BatchNormalization(name = "batch_norm7")(conv)
pooling = MaxPooling2D(pool_size=(2,1), name = "max_pool3")(bn)

conv = Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv8")(pooling)
bn = BatchNormalization(name = "batch_norm8")(conv)
conv = Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv9")(bn)
bn = BatchNormalization(name = "batch_norm9")(conv)
conv = Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv10")(bn)
bn = BatchNormalization(name = "batch_norm10")(conv)
pooling = MaxPooling2D(pool_size=(2,1), name = "max_pool4")(bn)

conv = Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv11")(pooling)
bn = BatchNormalization(name = "batch_norm11")(conv)
conv = Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv12")(bn)
bn = BatchNormalization(name = "batch_norm12")(conv)
conv = Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', name = "conv13")(bn)
bn = BatchNormalization(name = "batch_norm13")(conv)
pooling = MaxPooling2D(pool_size=(2,1), name = "max_pool5")(bn)

flatten = Flatten(name = "flatten")(pooling)

dense = Dense(5, name = "dense")(flatten)
y_pred = Activation('softmax', name='label')(dense)

Model(inputs = input_data, outputs = y_pred).summary()

#%%
filepath="Checkpoints/weights.1069e-0.63a-6.02l.hdf5"

model_p = Model(inputs=input_data, outputs=y_pred)
model_p.load_weights(filepath)

#%%
def read_one_image(filename):
    image = cv2.imread(filename)
    h, w, f = np.shape(image)
    
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    h, w, f = np.shape(image)
    
    return image

#%%
test = pd.read_csv("Data/test_ApKoW4T.csv")

image_out = np.zeros((len(test), height, width, 3))

for i in range(0, len(test)):
    path = "Data/Image/" + test["image"][i]
    print(path)
    image_out[i] = read_one_image(path)


#%%
arr = model_p.predict(image_out)
predx = np.argmax(arr, 1)

#%%
test["category"] = predx

test.to_csv("Data/Submission1.csv", index = False)
