# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:08:12 2019

@author: tornado
"""

import pandas as pd

from keras.layers import Input, TimeDistributed, Bidirectional, Conv2D, BatchNormalization, MaxPooling2D, Flatten, LSTM, Dense, Lambda, GRU, Activation
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.merge import add, concatenate

import keras.backend as K
#from time_distributed_read_images import generate_arrays_from_file, generate_val_arrays_from_file
from read_images import generate_arrays_from_file, generate_val_arrays_from_file

from config import height, width, train_file, mini_batch_size

from sklearn.model_selection import train_test_split

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
model = Model(inputs = input_data, outputs = y_pred)

# clipnorm seems to speeds up convergence
#opt = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
opt = Adam(lr = 0.0005)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

model.load_weights("gdrive/My Drive/Colab/Ship/Checkpoints/td.weights.last.hdf5")

#%%
#from keras.utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True)

#%%
data = pd.read_csv(train_file)
training, validation = train_test_split(data, test_size = 0.10)

training = training.reset_index().drop(columns = ["index"])
validation = validation.reset_index().drop(columns = ["index"])

roundofffortraining = (len(training) // mini_batch_size) * mini_batch_size
roundoffforvalidation = (len(training) // mini_batch_size) * mini_batch_size

training = training[:roundofffortraining]
validation = validation[:roundoffforvalidation]

print("Training on ", str(len(training)), " samples")
print("Validating on ", str(len(validation)), " samples")

print(training.head())

#%%
filepath="gdrive/My Drive/Colab/Ship/Checkpoints/weights.{acc:.3f}a-{loss:.3f}l.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

checkpoint_all = ModelCheckpoint("gdrive/My Drive/Colab/Ship/Checkpoints/td.weights.last.hdf5", monitor='loss', verbose=1, save_best_only=False)

traininglog = CSVLogger("gdrive/My Drive/Colab/Ship/Checkpoints/logs.csv", separator=',', append=True)

tensorboard = TensorBoard(log_dir="gdrive/My Drive/Colab/Ship/Checkpoints")

callbacks_list = [checkpoint, checkpoint_all, traininglog, tensorboard]

#%%
model.fit_generator(generator = generate_arrays_from_file(training), 
                    steps_per_epoch=len(training) // mini_batch_size, 
                    epochs=3000, 
                    callbacks=callbacks_list,
                    validation_data = generate_val_arrays_from_file(validation),
                    validation_steps=100,
                    initial_epoch=0
                    )