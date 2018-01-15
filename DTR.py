import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataset = pd.read_csv('Position_Salaries.csv')
X = Dataset.iloc[:,1:2].values
Y = Dataset.iloc[:,2:3].values

#Fitting dataset to DTR
from sklearn.tree import DecisionTreeRegressor
DTRegressor = DecisionTreeRegressor(random_state = 0)
DTRegressor.fit(X,Y)

DTRegressor.predict(6)
DTRegressor.predict(6.5)
DTRegressor.predict(6.8)
DTRegressor.predict(7)

#plotting the values
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color = 'red')
plt.plot(X_grid, DTRegressor.predict(X_grid), color = 'blue')
plt.show()



