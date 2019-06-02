# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:55:33 2019

@author: Gireesh Sundaram
"""

import cv2
import numpy as np
from config import width, height, mini_batch_size
from sklearn.preprocessing import OneHotEncoder

#%%
def read_one_image(filename):
    image = cv2.imread("gdrive/My Drive/Colab/Ship/Data/Image/" + filename)
    h, w, f = np.shape(image)
    
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    h, w, f = np.shape(image)
    
    return image

#%%
def OneHot(dataframe):
    OHE = OneHotEncoder().fit(np.asarray(dataframe["category"]).reshape(-1, 1))
    return OHE

#%%
def generate_arrays_from_file(infile):
    OHE = OneHot(infile)
    while True:
        for record in range(0, len(infile), mini_batch_size):
            mini_batch = 0
            image_out = np.zeros((mini_batch_size, height, width, 3))
            label_out = np.zeros((mini_batch_size, 5))
                
            while mini_batch < mini_batch_size:
                filename = infile["image"][record]
    
                image_out[mini_batch] = read_one_image(filename)
                label_out[mini_batch] = (OHE.transform(np.asarray(infile["category"][record]).reshape(-1,1))).toarray()
                
                mini_batch = mini_batch + 1
                record = record + 1
                
            inputs = {
                    'input_1': image_out     #Batch Size, h, w, no of channels
                    }
            outputs = {'label': label_out}    #Batch size, 1
            
            yield(inputs, outputs)
            
#%%
def generate_val_arrays_from_file(infile):
    OHE = OneHot(infile)
    while True:
        for record in range(0, len(infile)):
            image_out = np.zeros((1, height, width, 3))
            label_out = np.zeros((1, 5))      
            filename = infile["image"][record]
            image_out[0] = read_one_image(filename)
            label_out[0] = (OHE.transform(np.asarray(infile["category"][record]).reshape(-1,1))).toarray()
                
            inputs = {
                    'input_1': image_out     #Batch Size, h, w, no of channels
                    }
            outputs = {'label': label_out}    #Batch size, 1
            
            yield(inputs, outputs)            