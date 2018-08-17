# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:52:27 2018

@author: Vinayak_S
"""
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing data
df = pd.read_csv('Data.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#Filling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
X[:,0] = LabelEncoder().fit_transform(X[:,0])

from sklearn.preprocessing import OneHotEncoder
X = OneHotEncoder(categorical_features=[0]).fit_transform(X).toarray()

y = LabelEncoder().fit_transform(y.reshape(-1,1))
y.shape

#Splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
X_mod = StandardScaler().fit_transform(X[:,3:5])