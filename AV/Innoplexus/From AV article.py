# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:37:39 2018

@author: Gireesh Sundaram
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk import wordpunct_tokenize

#%%
input_file = pd.read_csv("D:\\Hackathons\\Innoplexus\\Data\\train.csv")
input_file["Together"] = input_file["Domain"] + " " + input_file["Url"]
input_file = input_file.drop(["Domain", "Url"], axis = 1)

docs1 = input_file['Together'].tolist()

input_list = []

for items in range(0, len(docs1)):
    punt = wordpunct_tokenize(docs1[items])
    punt = [words for words in punt  if len(words) > 3]
    string1 = ""
    for words in punt:
        string1 = string1 + " " + words
    input_list.append(string1)
#%%
train_x, valid_x, train_y, valid_y = train_test_split(input_list, input_file["Tag"])

encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

#%%
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(input_list)

train_x_count = count_vect.transform(train_x)
valid_x_count = count_vect.transform(valid_x)

#%%
BNB = BernoulliNB()
BNB.fit(train_x_count,train_y)
prediction = BNB.predict(valid_x_count)
accuracy_score(valid_y, prediction)

#%%
LR = LogisticRegression()
LR.fit(train_x_count,train_y)
prediction = LR.predict(valid_x_count)
accuracy_score(valid_y, prediction)

#%%
submission = pd.read_csv("D:\\Hackathons\\Innoplexus\\Data\\test_nvPHrOx.csv")
submission["Together"] = submission["Domain"] + " " + submission["Url"]

docs1 = submission['Together'].tolist()

submission_list = []

for items in range(0, len(docs1)):
    punt = wordpunct_tokenize(docs1[items])
    punt = [words for words in punt  if len(words) > 3]
    string1 = ""
    for words in punt:
        string1 = string1 + " " + words
    submission_list.append(string1)

submission_count = count_vect.transform(submission_list)

prediction = LR.predict(submission_count)
submission["Tag"] = encoder.inverse_transform(prediction)

submission = submission[["Webpage_id", "Tag"]]

submission.to_csv("D:\\Hackathons\\Innoplexus\\Data\\submission6.csv", index=False)
