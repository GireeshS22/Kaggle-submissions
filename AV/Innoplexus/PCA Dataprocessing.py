# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 09:48:13 2018
@author: Gireesh Sundaram
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

#%%
data = pd.read_csv("D:\\Hackathons\\Promotion\\train_LZdllcl.csv")

#%%
data.isnull().sum()
data["education"] = data["education"].fillna("Unknown")
data["previous_year_rating"] = data["previous_year_rating"].fillna(3.0)

data["education"] = np.where(data["age"] > 35, "Does not matter", data["education"])
data["education"] = np.where(data["length_of_service"] > 5, "Does not matter", data["education"])

data["recruitment_channel"] = np.where(data["length_of_service"] > 3, "Does not matter", data["recruitment_channel"])

#%%
cat_columns = data.select_dtypes(include=[object])
LE = LabelEncoder()
cat_columns_1 = cat_columns.apply(LE.fit_transform)

#%%
enc = OneHotEncoder()
hotlab = enc.fit_transform(cat_columns_1).toarray()
hotlab = pd.DataFrame(hotlab)

#%%
data= data.drop(columns = ["employee_id"])

#%%
corelation = data.corr()

#%%
cols = ["previous_year_rating", "awards_won?", "KPIs_met >80%", "avg_training_score", 'no_of_trainings', 'age']

#%%
for col in cols:
    hotlab[col] = data[col]

promoted = data[["is_promoted"]]

#%%
x_train, x_test, y_train, y_test = train_test_split(hotlab, promoted)

sm = SMOTE(random_state=20)

train_input_new, train_output_new = sm.fit_sample(x_train, y_train)

#%%
class1 = ExtraTreeClassifier()
class1.fit(x_train, y_train)
pred1 = class1.predict(x_test)
score = f1_score(y_test, pred1)

#%%

confussion = confusion_matrix(y_test, pred1)

#%%
#For submission
submission_data = pd.read_csv("D:\\Hackathons\\Promotion\\test_2umaH9m.csv")

#%%
submission_data["education"] = submission_data["education"].fillna("Unknown")
submission_data["previous_year_rating"] = submission_data["previous_year_rating"].fillna(np.mean(submission_data["previous_year_rating"]))

submission_data["education"] = np.where(submission_data["age"] > 35, "Does not matter", submission_data["education"])
submission_data["education"] = np.where(submission_data["length_of_service"] > 5, "Does not matter", submission_data["education"])

submission_data["recruitment_channel"] = np.where(submission_data["length_of_service"] > 3, "Does not matter", submission_data["recruitment_channel"])

cat_columns = submission_data.select_dtypes(include=[object])
cat_columns_1 = cat_columns.apply(LE.fit_transform)
hotlab_out = enc.fit_transform(cat_columns_1).toarray()
hotlab_out = pd.DataFrame(hotlab_out)

for col in cols:
    hotlab_out[col] = submission_data[col]

#%%
class1.fit(hotlab, promoted)

final_submission = pd.DataFrame(submission_data["employee_id"])
final_submission ["is_promoted"] = class1.predict(hotlab_out)
final_submission.to_csv("D:\\Hackathons\\Promotion\\submission_14_GB.csv", index = False)