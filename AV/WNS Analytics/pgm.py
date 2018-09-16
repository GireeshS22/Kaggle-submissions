# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:14:14 2018

@author: Gireesh Sundaram
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#%%
data = pd.read_csv("D:\\Hackathons\\Promotion\\train_LZdllcl.csv")
data.shape
data.info()
data.describe()

#%%
g = sns.FacetGrid(data, col = "is_promoted")
g.map(plt.hist, "avg_training_score")


#%%
pd.crosstab(data["gender"], data["is_promoted"]).plot(kind='bar')

#%%
numeric_columns = ["no_of_trainings", "age", "length_of_service", "avg_training_score"]

data["education"] = np.where(data["age"] > 40, "Does not matter", data["education"])
data["education"] = np.where(data["length_of_service"] > 5, "Does not matter", data["education"])

data["recruitment_channel"] = np.where(data["length_of_service"] > 5, "Does not matter", data["recruitment_channel"])

data.isnull().sum(axis = 0)
data["education"] = data["education"].fillna("Missing")
data["previous_year_rating"] = data["previous_year_rating"].fillna(3.0)

#%%
#data = data[(np.abs(stats.zscore(data[numeric_columns])) < 3).all(axis=1)]

#%%
#Changing the departments
data["department"] = np.where(data["department"].isin(['Finance', 'HR', 'Legal']), "Operational", data["department"])
data["department"] = np.where(data["department"].isin(['Analytics']), "Technology", data["department"])

#%%
processed_data = data[numeric_columns]
processed_data["department"] = LabelEncoder().fit(data["department"]).transform(data["department"])
processed_data["region"] = LabelEncoder().fit(data["region"]).transform(data["region"])
processed_data["education"] = LabelEncoder().fit(data["education"]).transform(data["education"])
processed_data["gender"] = LabelEncoder().fit(data["gender"]).transform(data["gender"])
processed_data["recruitment_channel"] = LabelEncoder().fit(data["recruitment_channel"]).transform(data["recruitment_channel"])
processed_data["is_promoted"] = data["is_promoted"]
processed_data["previous_year_rating"] = data["previous_year_rating"]
processed_data["length_of_service"] = data["length_of_service"]
processed_data["awards_won?"] = data["awards_won?"]
processed_data["KPIs_met"] = data["KPIs_met >80%"]

#%%
result = processed_data["is_promoted"]
processed_data = processed_data.drop(columns = ["is_promoted"])

#%%
x_train, x_test, y_train, y_test = train_test_split(processed_data, result)

#%%
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

prediction = model.predict(x_test)
score = f1_score(y_test, prediction)

#%%
#For submission
submission_data = pd.read_csv("D:\\Hackathons\\Promotion\\test_2umaH9m.csv")

#%%
submission_data["education"] = np.where(submission_data["age"] > 40, "Does not matter", submission_data["education"])
submission_data["education"] = np.where(submission_data["length_of_service"] > 10, "Does not matter", submission_data["education"])

submission_data["recruitment_channel"] = np.where(submission_data["length_of_service"] > 5, "Does not matter", submission_data["recruitment_channel"])

submission_data["education"] = submission_data["education"].fillna("Missing")
submission_data["previous_year_rating"] = submission_data["previous_year_rating"].fillna(3.0)
sub_pd = submission_data[numeric_columns]

#%%

submission_data["department"] = np.where(submission_data["department"].isin(['Finance', 'HR', 'Legal']), "Operational", submission_data["department"])
submission_data["department"] = np.where(submission_data["department"].isin(['Analytics']), "Technology", submission_data["department"])

sub_pd["department"] = LabelEncoder().fit(data["department"]).transform(submission_data["department"])
sub_pd["region"] = LabelEncoder().fit(data["region"]).transform(submission_data["region"])
sub_pd["education"] = LabelEncoder().fit(data["education"]).transform(submission_data["education"])
sub_pd["gender"] = LabelEncoder().fit(data["gender"]).transform(submission_data["gender"])
sub_pd["recruitment_channel"] = LabelEncoder().fit(data["recruitment_channel"]).transform(submission_data["recruitment_channel"])
sub_pd["previous_year_rating"] = submission_data["previous_year_rating"]
sub_pd["length_of_service"] = submission_data["length_of_service"]
sub_pd["awards_won?"] = submission_data["awards_won?"]
sub_pd["KPIs_met"] = submission_data["KPIs_met >80%"]

#%%
model.fit(processed_data, result)
final_submission = pd.DataFrame(submission_data["employee_id"])
final_submission ["is_promoted"] = model.predict(sub_pd)
final_submission.to_csv("D:\\Hackathons\\Promotion\\submission_08_DT.csv", index = False)
