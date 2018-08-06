# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:55:49 2018

@author: Gireesh Sundaram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

#%%
train = pd.read_csv("D:\\Kaggle\\Santander\\Data\\train.csv")

columns = train.columns.tolist()
columns.remove("ID")
columns.remove("target")

columns = np.asarray(columns)

x = train.loc[:, columns].values

#%%
std_values = StandardScaler().fit_transform(x)
pca = PCA(n_components=4000)

principleComponents = pca.fit_transform(std_values)

#%%
variance = pca.explained_variance_ratio_
variance_ratio = np.cumsum(np.round(variance, decimals=10)*100)

#%%
plt.plot(variance_ratio)

#%%
TakenPCA = pd.DataFrame(principleComponents[:,:1210])
#TakenPCA["ID"] = train["ID"]
#TakenPCA["target"] = train["target"]

#%%
train_x = TakenPCA[:4000]
train_y = train["target"][:4000]

test_x = TakenPCA[4000:]
test_y = np.asarray(train["target"][4000:])

#%%
LR = LinearRegression()
LR.fit(train_x, train_y)

prediction = LR.predict(test_x)

LR_RMSE = np.sqrt(mean_squared_error(test_y, prediction))
LR_RMSE

#%%
Ridge = Ridge()
Ridge.fit(train_x, train_y)

prediction = LR.predict(test_x)

Ridge_RMSE = np.sqrt(mean_squared_error(test_y, prediction))
Ridge_RMSE

#%%
las = Lasso(alpha = 0.2)
las.fit(train_x, train_y)

prediction = LR.predict(test_x)

las_RMSE = np.sqrt(mean_squared_error(test_y, prediction))
las_RMSE
