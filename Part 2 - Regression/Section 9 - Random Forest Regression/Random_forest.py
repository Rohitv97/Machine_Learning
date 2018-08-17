# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:12:46 2018

@author: Vinayak_S
"""

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading data 
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
y = df.iloc[:,-1].values

#Implementing Random Forest Method
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)

#Predicting using Random Forest Regression
y_pred = regressor.predict(6.5)

#Visualising the result
X_grid = np.arange(min(X),max(X),0.01).reshape(-1,1)
plt.scatter(X,y)
plt.plot(X_grid, regressor.predict(X_grid))
plt.show()