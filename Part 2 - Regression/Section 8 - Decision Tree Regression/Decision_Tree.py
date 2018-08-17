# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:11:49 2018

@author: Vinayak_S
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading data
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

#Implementing Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#Predicting using Decision Tree Regressor
y_pred = regressor.predict(6.5)

#Visualisation of the Model
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1,1)
plt.scatter(X,y)
plt.plot(X_grid,regressor.predict(X_grid))
plt.title('Level vs Salary(Decision Tree Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')