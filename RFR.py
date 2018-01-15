#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

#build Regression model
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators = 100,random_state = 0)
RF_regressor.fit(X,Y)
#prediction
RF_regressor.predict(6.5)
#plotting 
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,RF_regressor.predict(X_grid),color = 'blue')
plt.show()

#build Regression model increasing N =200
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators = 200,random_state = 0)
RF_regressor.fit(X,Y)
#prediction
RF_regressor.predict(6.5)
#plotting 
X_grid = np.arange(min(X),max(X),0.05)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,RF_regressor.predict(X_grid),color = 'blue')
plt.show()

#build Regression model increasing N =300
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators = 300,random_state = 0)
RF_regressor.fit(X,Y)
#prediction￼
RF_regressor.predict(6.5)
#plotting 
X_grid = np.arange(min(X),max(X),0.05)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,RF_regressor.predict(X_grid),color = 'blue')
plt.show()

