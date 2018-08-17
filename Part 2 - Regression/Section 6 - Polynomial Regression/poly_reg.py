# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:41:08 2018

@author: Vinayak_S
"""

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#loading data
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values


#Transforming to Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly_scale = PolynomialFeatures(degree = 4)
X_trans = poly_scale.fit_transform(X)

#Applying Linear Regression
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_trans, y)

#Visualisation of the model
X_grid = np.arange(start = 1, stop = 10 , step = 0.01).reshape(-1,1)
plt.scatter(X, y)
plt.plot(X_grid, linear.predict(poly_scale.fit_transform(X_grid)))

#Prediction
X_test_trans = poly_scale.fit_transform(6.5)
linear.predict(X_test_trans)