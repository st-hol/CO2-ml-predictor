import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data from a CSV file
df = pd.read_csv('FuelConsumption.csv')  # Replace 'your_data.csv' with the actual file path

# Separate the features (X) and the target variable (y)
X = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]  # Replace with your actual feature column names
y = df['CO2EMISSIONS']  # Replace with your actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
degree = 2  # Задайте ступінь полінома
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train a linear regression model
poly_regressor = LinearRegression()
poly_regressor.fit(X_train_poly, y_train)

# Make predictions on the test set
poly_predictions = poly_regressor.predict(X_test_poly)

# Calculate MSE
poly_mse = mean_squared_error(y_test, poly_predictions)

# Calculate MAE
poly_mae = mean_absolute_error(y_test, poly_predictions)

# Calculate R-squared
poly_r2 = r2_score(y_test, poly_predictions)

print("Polynomial Regression (Degree={}):".format(degree))
print("Mean Squared Error (MSE):", poly_mse)
print("Mean Absolute Error (MAE):", poly_mae)
print("R-squared:", poly_r2)