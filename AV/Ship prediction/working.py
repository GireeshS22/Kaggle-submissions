# -*- coding: utf-8 -*-
"""
Created on Sat May 25 12:07:43 2019

@author: Gireesh Sundaram
"""

from sklearn.preprocessing import OneHotEncoder

#%%
label_out = OneHotEncoder().fit_transform(training["category"][5])