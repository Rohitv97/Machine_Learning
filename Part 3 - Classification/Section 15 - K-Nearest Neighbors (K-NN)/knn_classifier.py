# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 00:06:18 2018

@author: Vinayak_S
"""

#Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading data
df = pd. read_csv('Social_Network_aDS.csv')
X = df.iloc[:,2:-1].values
y = df.iloc[:,-1].values

#Splitting Data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Implementing Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
classifier.fit(X_train, y_train)

#Predicting Results
y_pred = classifier.predict(X_test)

#Comparing the accuracy of the model using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Visualisation of the results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train , y_train
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min() - 1, X_set[:,0].max() + 1, 0.01),
                     np.arange(X_set[:,1].min() - 1, X_set[:,1].max() + 1, 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red','green')))
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red','green'))(i))
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('KNeighbors Plot')
plt.legend()
plt.show()
    
#Visulaising test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test , y_test
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min() - 1, X_set[:,0].max() + 1, 0.01),
                     np.arange(X_set[:,1].min() - 1, X_set[:,1].max() + 1, 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red','green')))
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red','green'))(i), label = j)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('KNeighbors Plot')
plt.legend()
plt.show()