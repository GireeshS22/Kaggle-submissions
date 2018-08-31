# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 08:55:13 2018

@author: Gireesh Sundaram
"""

import pandas as pd

#%%
data = pd.read_csv("D:\\Kaggle\\NYTaxi\\Data\\train.csv")

#%%
sample500 = data.sample(n = 500000)
sample50 = sample500.sample(n = 50000)
sample5 = sample50.sample(n = 5000)

#%%
sample5.to_csv("D:\\Kaggle\\NYTaxi\\Data\\sample5.csv", index = False)
sample50.to_csv("D:\\Kaggle\\NYTaxi\\Data\\sample50.csv", index = False)
sample500.to_csv("D:\\Kaggle\\NYTaxi\\Data\\sample500.csv", index = False)
