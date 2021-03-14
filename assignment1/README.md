Assignment 1
In this assigment, you will be building models to predict house prices using the Boston housing dataset.

I - Linear Regression
Build a model using a linear regression (Scikit-learn) algorithm to predict house prices. You can pick a feature from the dataset to work with the model.
from sklearn.linear_model import LinearRegression
Y = C + w * X

Plot the data with the best fit line.
Calculate a RMSE score.
Calculate a R-squared score.
II - Polynomial Regression
Build a model using a Polynomial regression algorithm to predict house prices. Keep the same feature you selected from the previous part to work with the polynomial model.
Y = C + w1 * X + w2 * X2

from sklearn.preprocessing import PolynomialFeatures
X2 is only a feature, but the curve that we are fitting is in quadratic form.
Plot the best 2nd degree polynomail curve.
Calculate a RMSE score.
Calculate a R-squared score.
Plot another diagram for degree=20.
III - Multiple Regression
Build a model using a multiple regression algorithm to predict house prices. Select 3 or more features to work with the model.
Y = C + w1 * X1 + w2 * X2 + w3 * X3

Calculate a RMSE score.
Calculate a R-squared score.
Calculate an adjusted R-squared score.
