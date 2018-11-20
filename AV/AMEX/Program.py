# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 00:13:05 2018

@author: Gireesh Sundaram
"""

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

from imblearn.over_sampling import SMOTE

import xgboost as xgb

#%%
data = pd.read_csv("D:\\Hackathons\\Amex\\Datasets\\train.csv")
test = pd.read_csv("D:\\Hackathons\\Amex\\Datasets\\test.csv")
train = data.sample(frac = 0.9)

historic = pd.read_csv("D:\\Hackathons\\Amex\\Datasets\\historic_restruct.csv")

#%%
train['hour'] = pd.to_numeric(train['DateTime'].str.slice(11,13))
train["time"] = np.where(train['hour'].between(0, 4), "Midnight",
     np.where(train['hour'].between(5, 8), "Early Morning", 
              np.where(train['hour'].between(9, 12), "Morning", 
                       np.where(train['hour'].between(13, 16), "Afternoon", 
                                np.where(train['hour'].between(17, 20), "Evening", "Night")))))

#%%
train = train.merge(historic, on = ['user_id', 'product'], how='left')

interest_view = train[['view', 'interest']]
interest_view = interest_view.fillna(value = 0)

#%%
selectedfeatures = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'gender',  'user_group_id', 'age_level', 'user_depth']
selectedcols = train[selectedfeatures]

#%%
#Tryig to see if some row has any of the missing values, but does not!
navaluecols = ['user_group_id', 'age_level', 'user_depth', 'city_development_index']
handlingna = data[navaluecols]
handlingna["user_id"] = train["user_id"]
handlingna = handlingna.drop_duplicates()
user_id = handlingna[handlingna["user_id"].duplicated(keep=False)]

#%%
selectedcols['gender'] = selectedcols['gender'].fillna(value = "Female")
selectedcols['age_level'] = selectedcols['age_level'].fillna(value = 2)
selectedcols['user_depth'] = selectedcols['user_depth'].fillna(value = 1)
#selectedcols['city_development_index'] = selectedcols['city_development_index'].fillna(value = 3)

selectedcols = selectedcols.fillna(value = -99)
LE = LabelEncoder()
selectedcols_1 = selectedcols.apply(LE.fit_transform)

#%%
OHE = OneHotEncoder()
selectedcols_2 = OHE.fit_transform(selectedcols_1).toarray()
selectedcols_2 = pd.DataFrame(selectedcols_2)
selectedcols_2['is_click'] = train['is_click'].reset_index(drop=True)

#selectedcols_2['interest'] = interest_view['interest']
#selectedcols_2['view'] = interest_view['view']

#%%
x_train, x_test, y_train, y_test = train_test_split(selectedcols_2.drop(columns = ['is_click']), selectedcols_2['is_click'])

sm = SMOTE()
train_ip_new, train_op_new = sm.fit_sample(x_train, y_train)

#%%
model = DecisionTreeClassifier()
model.fit(train_ip_new, train_op_new)
prediction = model.predict(x_test)
score = f1_score(y_test, prediction)
recall = recall_score(y_test, prediction)
precision = precision_score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

#%%
def featureselection(dataframe):
    dataframe['hour'] = pd.to_numeric(dataframe['DateTime'].str.slice(11,13))
    selectedcols = dataframe[selectedfeatures]    
    selectedcols['gender'] = selectedcols['gender'].fillna(value = "Female")
    selectedcols['age_level'] = selectedcols['age_level'].fillna(value = 2)
    selectedcols['user_depth'] = selectedcols['user_depth'].fillna(value = 1)
    #selectedcols['city_development_index'] = selectedcols['city_development_index'].fillna(value = 3)
    selectedcols = selectedcols.fillna(value = -99)
    selectedcols_1 = selectedcols.apply(LE.fit_transform)
    selectedcols_2 = OHE.fit_transform(selectedcols_1).toarray()
    selectedcols_2 = pd.DataFrame(selectedcols_2)
    return selectedcols_2

#%%
preprocessed = featureselection(test)
output = model.predict(preprocessed)

#%%
final_submission = pd.DataFrame()
final_submission["session_id"] = test['session_id']
final_submission["is_click"] = output
final_submission.to_csv("D:\\Hackathons\\Amex\\Datasets\\submission_10_DT_improving_features.csv", index = False)

#%%
for items in selectedfeatures:
    print(items)
    print(data[items].unique())
    print(test[items].unique())
    
#%%
time_by_day = train[["hour", 'is_click']].groupby(["hour"]).sum()
count_gender = data.groupby(['product', 'gender']).size().reset_index(name='count')
count_age = data.groupby(['product', 'age_level']).size().reset_index(name='count')
count_depth = data.groupby(['product', 'user_depth']).size().reset_index(name='count')
count_city = data.groupby(['product', 'city_development_index']).size().reset_index(name='count')

#%%
interest = pd.read_csv("D:\\Hackathons\\Amex\\Datasets\\historical_user_logs.csv")

#%%
view = interest.groupby(['user_id', 'product', 'action']).size().reset_index(name='count')
view_p = view.pivot_table(index = ['user_id', 'product'], columns = 'action', values = 'count').reset_index().fillna(value = 0)
view_p.to_csv("D:\\Hackathons\\Amex\\Datasets\\historic_restruct.csv", index = False)

preprocessed.to_csv("D:\\Hackathons\\Amex\\Datasets\\preprocessed_op.csv", index = False)
