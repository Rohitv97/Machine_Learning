# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:35:46 2018

@author: Vinayak_S
"""

#importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#reading data
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

#Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#Fitting Linear Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

#Predicting the result
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#Visualisation for Training set
plt.scatter (X_train, y_train, color = 'red')
plt.plot(X_train, y_train_pred, color = 'blue')
plt.title('Years of experience vs Salary (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary in $')
plt.show()

#Visualisation for Testing set
plt.scatter (X_test,y_test,color = 'red')
plt.plot(X_test,y_test_pred,'blue')
plt.title('Years of experience vs Salary (Testing set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary in $')
plt.show()