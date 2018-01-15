# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 16:37:11 2017

@author: sadhana reddy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,-1].values

#plot the scatter plot of dataset to understand what model needs to be built for this dataset
plt.scatter(X,Y,color = 'red')
plt.show()

#creating linear regression model
from sklearn.linear_model import LinearRegression
LReg = LinearRegression()
LReg.fit(X, Y)
#plot the scatter plot of dataset to understand what model needs to be built for this dataset
plt.scatter(X,Y,color = 'red')
plt.plot(X,LReg.predict(X) , color = 'blue')
plt.show()

#Adding polynomial terms to the dataset
from sklearn.preprocessing import PolynomialFeatures
PolyReg = PolynomialFeatures(degree =4)
X_poly = PolyReg.fit_transform(X)
PReg = LinearRegression()
PReg.fit(X_poly, Y)
#plot the scatter plot of dataset to understand poly model needs to be built for this dataset
plt.scatter(X,Y,color = 'red')
plt.plot(X,PReg.predict(X_poly) , color = 'blue')
plt.show()

# to improve plot more, we could increase grain level to 0.1 steps
X_grid = np.arange(min(X),max(X),0.1) #this gives vector
X_grid = X_grid.reshape((len(X_grid)),1) #converting vector to matrix
plt.scatter(X,Y,color = 'red')
plt.plot(X_grid,PReg.predict(PolyReg.fit_transform(X_grid)) , color = 'blue')
plt.show()

#now prdict for new values
#using linear regression model
LReg.predict(6.5)
#using ploynomial regression model for same employee
PReg.predict(PolyReg.fit_transform(6.5))