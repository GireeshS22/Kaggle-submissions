# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:43:28 2018

@author: Gireesh Sundaram
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split

#%%
data = pd.read_csv("D:\\Hackathons\\Amex\\Datasets\\train.csv")
test = pd.read_csv("D:\\Hackathons\\Amex\\Datasets\\test.csv")
train = data.sample(frac = 0.2)

historic = pd.read_csv("D:\\Hackathons\\Amex\\Datasets\\historic_restruct.csv")

#%%
train['hour'] = pd.to_numeric(train['DateTime'].str.slice(11,13))
train["time"] = np.where(train['hour'].between(0, 4), "Midnight",
     np.where(train['hour'].between(5, 8), "Early Morning", 
              np.where(train['hour'].between(9, 12), "Morning", 
                       np.where(train['hour'].between(13, 16), "Afternoon", 
                                np.where(train['hour'].between(17, 20), "Evening", "Night")))))

#%%
train = train.merge(historic, on = ['user_id', 'product'], how='left')

interest_view = train[['view', 'interest']]
interest_view = interest_view.fillna(value = 0)

#%%
selectedfeatures = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'gender',  'user_group_id', 'age_level', 'user_depth']
selectedcols = train[selectedfeatures]

#%%
selectedcols['gender'] = selectedcols['gender'].fillna(value = "Female")
selectedcols['age_level'] = selectedcols['age_level'].fillna(value = 2)
selectedcols['user_depth'] = selectedcols['user_depth'].fillna(value = 3)
#selectedcols['city_development_index'] = selectedcols['city_development_index'].fillna(value = 3)

selectedcols = selectedcols.fillna(value = -99)
LE = LabelEncoder()
selectedcols_1 = selectedcols.apply(LE.fit_transform)

#%%
OHE = OneHotEncoder()
selectedcols_2 = OHE.fit_transform(selectedcols_1).toarray()
selectedcols_2 = pd.DataFrame(selectedcols_2)
selectedcols_2['is_click'] = train['is_click'].reset_index(drop=True)

#selectedcols_2['interest'] = interest_view['interest']
#selectedcols_2['view'] = interest_view['view']

#%%
x_train, x_test, y_train, y_test = train_test_split(selectedcols_2.drop(columns = ['is_click']), selectedcols_2['is_click'])

#%%
from keras.layers import Input, Embedding, Bidirectional, GlobalMaxPool1D, Dense, Dropout, CuDNNGRU
from keras.models import Model

#%%
embed_size = 300 # how big is each word vector
max_features = 200 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 60 # max number of words in a question to use

#%%
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#%%
model.fit(x_train, y_train, batch_size=512, epochs=2, validation_data=(x_test, y_test))
