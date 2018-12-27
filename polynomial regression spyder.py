import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')

# Always make sure x is matrix and y is vector
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

# Linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

# Linear regression visualization
plt.scatter(x,y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or bluff(linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Polynomial regression visualization
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg.predict(poly.fit_transform(x)), color = 'blue')
plt.title('Truth or bluff(polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Linear regression salary prediction
regressor.predict(5)

# Polynomial regression salary prediction
lin_reg.predict(poly.fit_transform(5))