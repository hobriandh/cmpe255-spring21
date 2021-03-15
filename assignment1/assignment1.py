import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import operator

#read data
data = "data/housing.csv"
variables = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv(data, delim_whitespace = True, names = variables)
prices = df['MEDV']
features = df.drop('MEDV', axis = 1)
print("Boston housing dataset has {} data points with {} variables each.".format(*df.shape))

#quick data analysis
minimum_price = np.amin(prices)
maximum_price = np.amax(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)
print("\nStatistics for Boston housing dataset:")
print("Minimum price: ${}".format(minimum_price * 1000)) 
print("Maximum price: ${}".format(maximum_price * 1000))
print("Mean price: ${}".format(mean_price * 1000))
print("Median price ${}".format(median_price * 1000))
print("Standard deviation of prices: ${}".format(std_price* 1000))

#finding correlation
correlation = df.corr()
f,ax=plt.subplots(figsize=(12,7))
sns.heatmap(correlation,cmap='viridis',annot=True)
plt.figure(1)
plt.title("Correlation between features",weight='bold', fontsize=18)
plt.show()

#breaking up the data
row, col = df.shape
x = df['RM'].to_numpy().reshape(row, 1)
y = df['MEDV'].to_numpy().reshape(row, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 13)

#part1
model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_train)

print("\nLinear Regression")
print("Using the feature RM")
print("Training Accuracy: ", r2_score(y_train, y_predict))
print("Root Mean Square: ", np.sqrt(mean_squared_error(y_train, y_predict)))
plt.figure(2)
plt.title("Linear Regression Model",weight='bold', fontsize=18)
plt.scatter(x_train, y_train)
plt.plot(x_train, y_predict)
plt.xlabel("Rooms")
plt.ylabel("Housing Price * $1000")
plt.show()

#part2
polynomial = PolynomialFeatures(degree=2)
model2 = make_pipeline(polynomial, model)
model2.fit(x_train, y_train)
y_predict2 = model2.predict(x_train)

print("\nPolynomial Regression 2nd Degree")
print("Using the feature RM")
print("Training Accuracy: ", r2_score(y_train, y_predict2))
print("Root Mean Square: ", np.sqrt(mean_squared_error(y_train, y_predict2)))

plt.figure(3)
plt.title("Polynomial Regression Model 2nd Degree",weight='bold', fontsize=18)
plt.scatter(x_train, y_train)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x_train, y_predict2), key = sort_axis)
x_train2, y_predict2 = zip(*sorted_zip)
plt.plot(x_train2, y_predict2, color = 'r')
plt.xlabel("Rooms")
plt.ylabel("Housing Price * $1000")
plt.show()

polynomial = PolynomialFeatures(degree=20)
model3 = make_pipeline(polynomial, model)
model3.fit(x_train, y_train)
y_predict3 = model3.predict(x_train)

print("\nPolynomial Regression 20th Degree")
print("Using the feature RM")
print("Training Accuracy: ", r2_score(y_train, y_predict3))
print("Root Mean Square: ", np.sqrt(mean_squared_error(y_train, y_predict3)))

plt.figure(4)
plt.title("Polynomial Regression Model 20th Degree",weight='bold', fontsize=18)
plt.scatter(x_train, y_train)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x_train, y_predict3), key = sort_axis)
x_train3, y_predict3 = zip(*sorted_zip)
plt.plot(x_train3, y_predict3, color = 'r')
plt.xlabel("Rooms")
plt.ylabel("Housing Price * $1000")
plt.show()

#part3
row, col = df.shape
x = df[['RM', 'PTRATIO', 'LSTAT']].to_numpy()
y = df[['MEDV']].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 13)

model4 = LinearRegression()
model4.fit(x_train, y_train)
y_predict4 = model4.predict(x_train)

print("\nMultiple Polynomial Regression")
print("Using the feature RM, PTRATIO, and LSTAT")
print("Training Accuracy: ", r2_score(y_train, y_predict4))
print("Adjusted R2 Score: ", 1 - (1 - r2_score(y_train, y_predict4)) * (len(x_train) - 1) / (len(x_train) - x_train.shape[1] - 1))
print("Root Mean Square: ", np.sqrt(mean_squared_error(y_train, y_predict4)))