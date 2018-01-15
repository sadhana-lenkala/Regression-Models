# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 23:24:24 2017

@author: sadhana reddy
"""

#import dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
Dataset = pd.read_csv('50_Startups.csv')
X = Dataset.iloc[:,:-1].values
Y = Dataset.iloc[:,-1].values

#creating dummy variable for categorical variable "State"
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Encode_state = LabelEncoder()
X[:,3] = Encode_state.fit_transform(X[:,3])
Onehotencoder = OneHotEncoder(categorical_features = [3])
X = Onehotencoder.fit_transform(X).toarray()

# remove dummy variables
X = X[:,1:]

#create train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.8,random_state = 0)

#creating MLR regression on train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# predicting values of test set on trained model
Y_pred = regressor.predict(X_test)

#Performing Backward Elimination
import statsmodels.formula.api as sm
# add new colums of 1's to add constant to the equation as it is missed in sm library
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#this can also be added as
# X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
#now that model with all variables is created, check fro p-values
regressor_OLS.summary()
# as p value is greater than alpha for x2, remove x2
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
#removing 1 st variable too
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
#removing 4th variable
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
#removing 5th variable p > 0.05
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()