# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 22:43:55 2018

@author: Vinayak_S
"""

#Impporting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading data
df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:,2:-1].values
y = df.iloc[:,-1].values

#Splitting data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting SVM Classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the results
y_pred = classifier.predict(X_test)

#Comparing the accuracy of the model using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Visualising the results for training set
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min() - 1, X_set[:,0].max() + 1, 0.01 ), 
                     np.arange(X_set[:,1].min() - 1, X_set[:,1].max() + 1, 0.01 ))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red','green')))
plt.xlim()
plt.ylim()
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1], 
                c = ListedColormap(('red','green'))(i), label = j)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.title('SVM Model')
plt.show()

#Visualising the results for test set
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min() - 1, X_set[:,0].max() + 1, 0.01 ), 
                     np.arange(X_set[:,1].min() - 1, X_set[:,1].max() + 1, 0.01 ))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red','green')))
plt.xlim()
plt.ylim()
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1], 
                c = ListedColormap(('red','green'))(i), label = j)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.title('SVM Model')
plt.show()
