# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 23:34:25 2017

@author: sadhana reddy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datsaet
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:1].values
Y = dataset.iloc[:,1].values

#splitting into train and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

#perform / fit SLR for train dataset
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train,Y_train)

#predict the test set results and compare
y_pred = Regressor.predict(X_test)

#plotting y_pred for x_train vs x_train
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,Regressor.predict(X_train),color='blue')
plt.title('Salary vs years of Experience')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()

#plotting y_pred for x_test vs x_test
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,Regressor.predict(X_train),color='blue')
plt.title('Salary vs years of Experience for test set')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()
