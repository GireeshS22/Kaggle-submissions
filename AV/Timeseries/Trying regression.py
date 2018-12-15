# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 10:28:45 2018

@author: Gireesh Sundaram
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression

import numpy as np

#%%
train = pd.read_csv("D:\\Hackathons\\Genpact\\Data\\train.csv")

#%%
#Prepraing features
TransformColumns = train[["center_id", "meal_id"]]
LE = LabelEncoder()
selectedcols_1 = TransformColumns.apply(LE.fit_transform)
OHE = OneHotEncoder()
selectedcols_2 = OHE.fit_transform(selectedcols_1).toarray()
selectedcols_2 = pd.DataFrame(selectedcols_2)

#%%
selectedcols_2['week'], selectedcols_2['base_price'], selectedcols_2['checkout_price'], selectedcols_2['emailer_for_promotion'], selectedcols_2['homepage_featured'] = train.week, train.base_price, train.checkout_price, train.emailer_for_promotion, train.homepage_featured

#%%
x_train, x_text, y_train, y_test = train_test_split(selectedcols_2, train['num_orders'])

#%%ac
model = LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_text)

prediction[prediction > 1000] = 1000
prediction[prediction < 0] = 1

RMSLE = np.sqrt(mean_squared_log_error(y_test, prediction))
print(RMSLE*100)
