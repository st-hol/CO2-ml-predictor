import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the data from a CSV file
df = pd.read_csv('FuelConsumption.csv')  # Replace 'your_data.csv' with the actual file path

# Separate the features (X) and the target variable (y)
X = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']]  # Replace with your actual feature column names
y = df['CO2EMISSIONS']  # Replace with your actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.211, random_state=42)

# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
lr_predictions = lr.predict(X_test)

# Виведення метрик
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lr_predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, lr_predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lr_predictions)))

# Побудова графіку порівняння прогнозованих і реальних значень
plt.scatter(y_test, lr_predictions)
plt.xlabel("Реальні значення")
plt.ylabel("Прогнозовані значення")
plt.title("Порівняння прогнозованих та реальних значень CO2 Emissions")
plt.show()



#1
plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS,  color='blue')
plt.xlabel("Об'єм двигуна")
plt.ylabel("Викиди СО2")
plt.title("Аналіз залежності викидів СО2 від об'єму двигуна")
plt.show()

# msk = np.random.rand(len(df)) < 0.8
# train = df[msk]
# test = df[~msk]
# regr = LinearRegression()
# train_x = np.asanyarray(train[['ENGINESIZE']])
# train_y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (train_x, train_y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)
# print ('Intercept: ',regr.intercept_)
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], 'red')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")

#2
plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS,  color='blue')
plt.xlabel("Розхід палива")
plt.ylabel("Викиди СО2")
plt.title("Аналіз залежності викидів СО2 від розходу палива")
plt.show()

#3
plt.scatter(df.CYLINDERS, df.CO2EMISSIONS, color='blue')
plt.xlabel("Кількість циліндрів")
plt.ylabel("Викиди СО2")
plt.title("Аналіз залежності викидів СО2 від кількості циліндрів")
plt.show()


viz = df[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()