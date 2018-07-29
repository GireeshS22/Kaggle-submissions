# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 12:55:47 2018

@author: Gireesh Sundaram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

#%%
wine = pd.read_csv("D:\\Kaggle\\Wine\\Data\\winequality-red.csv")

#%%
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']

standard_values = StandardScaler().fit_transform(wine)

#%%
kmc = KMeans(n_clusters = 3, max_iter=100)
kmc.fit_transform(standard_values)

wine["Clusters without PCA"] = kmc.labels_

#%%
PCA = PCA(n_components=12)
PrincipleComponents = PCA.fit_transform(standard_values)
variance = PCA.explained_variance_ratio_
variance_ratio = np.cumsum(np.round(variance, decimals=3)*100)
plt.plot(variance_ratio)

TakenPCA = PrincipleComponents[:,:5]

#%%
kmc.fit_transform(TakenPCA)

wine["Clusters PCA"] = kmc.labels_

#%%
ClusterMean_NonPCA = wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality', "Clusters without PCA"]].groupby("Clusters without PCA").mean().sort_values('fixed acidity').values


ClusterMean_PCA = wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality', "Clusters PCA"]].groupby("Clusters PCA").mean().sort_values('fixed acidity').values
                        
difference = ClusterMean_NonPCA - ClusterMean_PCA
RMSE = np.sqrt(mean_squared_error(ClusterMean_NonPCA, ClusterMean_PCA))
RMSE
