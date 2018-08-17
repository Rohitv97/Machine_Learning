# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:38:44 2018

@author: Vinayak_S
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading data
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
Sc_x = StandardScaler()
Sc_y = StandardScaler()
X = Sc_x.fit_transform(X)
y = Sc_y.fit_transform(y.reshape(-1,1))

#Creating SVR Model
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, y.ravel())

#Visualisation
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1,1)
plt.scatter(X,y)
plt.plot(X_grid, regressor.predict(X_grid))
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Prediction
y_pred = Sc_y.inverse_transform(regressor.predict(Sc_x.transform(6.5)))