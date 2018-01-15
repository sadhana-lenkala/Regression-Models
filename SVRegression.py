# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 18:43:45 2017

@author: sadhana reddy
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
Dataset = pd.read_csv('Position_Salaries.csv')
X = Dataset.iloc[:,1:2].values
Y = Dataset.iloc[:, 2:3].values

#feature scaling to lessen the value of degree3 varables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#filltig SVR to dataset
from sklearn.svm import SVR
SV_regressor = SVR(kernel = 'rbf')
SV_regressor.fit(X,Y)

#predict using SVR
y_pred = SV_regressor.predict(sc_X.transform(np.array([[2.5]])))
y_final = sc_Y.inverse_transform(y_pred)
#plotting results
plt.scatter(X,Y,color = 'red')
plt.plot(sc_X.fit_transform(X),SV_regressor.predict(X), color = 'blue')
plt.show()
