# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 02:49:22 2020

@author: admin
"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#from sklearn.svm import SGD

iris = datasets.load_iris()
X = iris["data"][:,(2,3)] # petel length and width
y = (iris["target"] == 2).astype(np.float64) # Iris virginica
#y = iris["target"]
#y = iris["target"] == 2 
'''
# Using LinearSVC
svm_clf = Pipeline([
        ("scalar", StandardScaler()),
        ("linear_svc",LinearSVC(C=1, loss = "hinge"))
        ])
svm_clf.fit(X,y)
svm_clf.predict([[5.5,5.7]])
'''

#Using SVC
#svm_svc = SVC(C=1,kernel='linear')#,gamma = 'auto')
svm_svc = Pipeline([
        ("scalar", StandardScaler()),
        ("linear_svc",SVC(C=1, kernel ="linear"))
        ])
svm_svc.fit(X,y)
svm_svc.predict([[5.5,5.7]])

'''
#Using Stochastic Gradient Descent
'''

