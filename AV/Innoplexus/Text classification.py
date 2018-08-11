# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 00:19:02 2018

@author: Gireesh Sundaram
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
import numpy as np

#%%
input_file = pd.read_csv("D:\\Hackathons\\Innoplexus\\Data\\train.csv")

input_file['Together'] = input_file['Domain'] + " " + input_file['Url']
docs2 = input_file['Together'].tolist()

submission = pd.read_csv("D:\\Hackathons\\Innoplexus\\Data\\test_nvPHrOx.csv")

submission['Together'] = submission['Domain'] + " " + submission['Url']
docs1 = submission['Together'].tolist()

labels = input_file['Tag'].tolist()
unique_labels = list(set(input_file['Tag'].tolist()))
mapping = {k: v for v, k in enumerate(unique_labels)}
input_file["Mapped"] = input_file["Tag"].map(mapping)
labels = input_file['Mapped'].tolist()

docs = docs2 + docs1

docs = CountVectorizer().fit_transform(docs)

#%%
train_docs = docs[:40000]
test_docs = docs[40000:53447]

train_labels = labels[:40000]
test_labels = labels[40000:]

#%%
NB = BernoulliNB().fit(train_docs, train_labels)
prediction = NB.predict(test_docs)

RMSE = accuracy_score(test_labels, prediction)
RMSE

#%%
submission_docs = docs[-25787:]
submission_predictions = NB.predict(submission_docs)

submission["Prediction"] = submission_predictions
mapping_rev = {val:key for (key, val) in mapping.items()}

submission["Tag"] = submission["Prediction"].map(mapping_rev)
submission = submission[["Webpage_id", "Tag"]]

submission.to_csv("D:\\Hackathons\\Innoplexus\\Data\\submission9.csv", index=False)
