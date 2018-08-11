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
train_x = TakenPCA
train_y = train["target"]
#%%
LR = LinearRegression()
LR.fit(train_x, train_y)

#%%
submission = pd.read_csv("D:\\Kaggle\\Santander\\Data\\test.csv")
submission = submission.fillna("0")
columns = submission.columns.tolist()
columns.remove("ID")

columns = np.asarray(columns)

x = submission.loc[:, columns].values

#%%
std_values = StandardScaler().fit_transform(x)
pca = PCA(n_components=1210)

principleComponents = pca.fit_transform(std_values)

#%%
TakenPCA = pd.DataFrame(principleComponents[:,:1210])

prediction = LR.predict(TakenPCA)
submission_file = pd.DataFrame(np.abs(prediction), columns = ["Target"])
submission_file["ID"] = submission["ID"]

submission_file = submission_file[["ID", "Target"]]

submission_file.to_csv("D:\\Kaggle\\Santander\\Data\\submission.csv", index=False)
