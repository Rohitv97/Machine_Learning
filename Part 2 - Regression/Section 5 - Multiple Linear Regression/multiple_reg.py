# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 23:21:04 2018

@author: Vinayak_S
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

def BackwardElimination(x,SL):
    numvars = len(x[0])
    for i in range(numvars):
        regressor = smf.OLS(endog = y, exog = x).fit()
        max_val = max(regressor.pvalues).astype(float)
        if (max_val > 0.05):
            for j in range (numvars - i):
                if (regressor.pvalues[j].astype(float) == max_val):
                    x = np.delete(x,j,1)
    print(regressor.summary())
    return x


df = pd.read_csv('50_Startups.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,3] = LabelEncoder().fit_transform(X[:,3])
X = OneHotEncoder(categorical_features=[3]).fit_transform(X).toarray()

#Avoiding dummy variable trap
X = X[:,1:]

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)

#Multivariable linear Regression 
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X_train,y_train)
y_test_pred = regressor.predict(X_test)

#Implementing Backward Elimination

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
X_opt = BackwardElimination(X_opt, 0.05)

