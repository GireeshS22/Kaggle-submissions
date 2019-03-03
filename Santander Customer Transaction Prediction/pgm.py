# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:16:43 2019

@author: Gireesh Sundaram
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

#%%
train_full = pd.read_csv("D:\Kaggle\Santander_classification\Data\\train.csv")

train = train_full.sample(n = 20000).reset_index()

#%%
features = train.drop(columns = ["index", "ID_code", "target"])

#%%
def preprocessing(dataframe):
    standardized = StandardScaler().fit_transform(dataframe)
    PrincipalComponent = PCA(n_components=199)
    PrincipleComp = PrincipalComponent.fit_transform(standardized)
    
    variance = PrincipalComponent.explained_variance_ratio_
    variance_ratio = np.cumsum(np.round(variance, decimals=10)*100)
    print(variance_ratio)
    return PrincipleComp

PrincipleComp = preprocessing(features)

#%%
output = train["target"]

#%%
x_train, x_test, y_train, y_test = train_test_split(PrincipleComp, output)

#%%
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf

#%%
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#%%
model = Sequential([
        Dense(256, input_dim = 150, kernel_initializer='normal', activation='relu'),
        Dropout(0.6),
        BatchNormalization(),
        Dense(64, kernel_initializer='normal', activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(16, kernel_initializer='normal', activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(4, kernel_initializer='normal', activation='tanh'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(1, kernel_initializer='normal', activation='sigmoid')
        ])

#%%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])
model.summary()

#%%
model.fit(x_train, y_train, batch_size=500, epochs = 10, validation_data=(x_test, y_test))

#%%
predictions = model.predict(x_test)
#predictions = (predictions > 0.5) * 1

score = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, predictions)

#%%
test_full = pd.read_csv("D:\Kaggle\Santander_classification\Data\\test.csv")
test = test_full.drop(columns = ["ID_code"])

test_features = preprocessing(test)

target = model.predict(test_features)
target = (target > 0.5) * 1

#%%
submission = pd.DataFrame()
submission["ID_code"], submission["target"] = test_full["ID_code"], target

submission.to_csv("D:\Kaggle\Santander_classification\Data\\submission1.csv", index = False)
