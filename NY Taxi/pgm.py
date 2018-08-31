# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 08:55:13 2018

@author: Gireesh Sundaram
"""

import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import numpy as np

#%%
data = pd.read_csv("D:\\Kaggle\\NYTaxi\\Data\\sample500.csv")
data.isnull().sum()
data = data.dropna(how = 'any', axis = 'rows')

#%%
data["pickup"] = pd.to_datetime(data["pickup_datetime"])
data["Day"] = data["pickup"].dt.weekday_name
data["Month"] = data["pickup"].dt.month
data["Hour"] = data["pickup"].dt.hour
data["lat_diff"] = (data["pickup_latitude"] - data["dropoff_latitude"]).abs()
data["long_diff"] = (data["pickup_longitude"] - data["dropoff_longitude"]).abs()
data = data.drop(data[data["lat_diff"] == 0].index)

for elem in data["Day"].unique():
    data[str(elem)] = (data["Day"] == elem)*1
    
for elem in data["Month"].unique():
    data["Month" + str(elem)] = (data["Month"] == elem)*1
    
for elem in data["Hour"].unique():
    data["Hour" + str(elem)] = (data["Hour"] == elem)*1    
    
#%%
for_regression = data.drop(columns = ["key", "pickup", "Day", "Month", "pickup_latitude",
                                      "dropoff_latitude", "pickup_longitude", "dropoff_longitude", "pickup_datetime",
                                      "Monday", "Month1", "Hour", "Hour0"])
        
for_regression.to_csv("D:\\Kaggle\\NYTaxi\\Data\\for_regression.csv", index = False)

fitToPCA = for_regression.drop(columns = ["fare_amount"])

pca = PCA(n_components = 43)
PrincipleComponents = pca.fit_transform(fitToPCA)

variance = pca.explained_variance_ratio_
variance_ratio = np.cumsum(np.round(variance, decimals=10)*100)

pca_df = pd.DataFrame(PrincipleComponents)
pca_df["fare_amount"] = for_regression["fare_amount"].values

pca_df.to_csv("D:\\Kaggle\\NYTaxi\\Data\\for_regression.csv", index = False)

#%%

pca_df = pca_df[[0,1,2,3,7,12,15,18,20,21,22,23,24,30,35,38,39,40,41,42]]

y = for_regression["fare_amount"].values
x = pca_df.values

#%%
train_x, valid_x, train_y, valid_y = train_test_split(x, y)

#%%
LR = LinearRegression()
LR.fit(train_x, train_y)

prediction = LR.predict(valid_x)

RMSE = np.sqrt(mean_squared_error(valid_y, prediction))
print(RMSE)

#%%
#This is for submission
submission = pd.read_csv("D:\\Kaggle\\NYTaxi\\Data\\test.csv")

#%%
def DataPrepForSubmission(dataframe):
    dataframe["pickup"] = pd.to_datetime(dataframe["pickup_datetime"])
    dataframe["Day"] = dataframe["pickup"].dt.weekday_name
    dataframe["Month"] = dataframe["pickup"].dt.month
    dataframe["Hour"] = dataframe["pickup"].dt.hour
    dataframe["lat_diff"] = (dataframe["pickup_latitude"] - dataframe["dropoff_latitude"]).abs()
    dataframe["long_diff"] = (dataframe["pickup_longitude"] - dataframe["dropoff_longitude"]).abs()
    
    for elem in dataframe["Day"].unique():
        dataframe[str(elem)] = (dataframe["Day"] == elem)*1
        
    for elem in dataframe["Month"].unique():
        dataframe["Month" + str(elem)] = (dataframe["Month"] == elem)*1

    for elem in dataframe["Hour"].unique():
        dataframe["Hour" + str(elem)] = (dataframe["Hour"] == elem)*1            
        
    for_regression = dataframe.drop(columns = ["key", "pickup", "Day", "Month", "pickup_latitude",
                                          "dropoff_latitude", "pickup_longitude", "dropoff_longitude", "pickup_datetime",
                                          "Monday", "Month1", "Hour", "Hour0"])
    
    pca = PCA(n_components = 43)
    PrincipleComponents = pca.fit_transform(for_regression)
        
    pca_df = pd.DataFrame(PrincipleComponents)
    pca_df = pca_df[[0,1,2,3,7,12,15,18,20,21,22,23,24,30,35,38,39,40,41,42]]
    return pca_df
    
#%%
one = DataPrepForSubmission(dataframe = submission).values
submission["fare_amount"] = LR.predict(one)

output = submission[["key", "fare_amount"]]
output.to_csv("D:\\Kaggle\\NYTaxi\\Data\\submission.csv", index =False)
