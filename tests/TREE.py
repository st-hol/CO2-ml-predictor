import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data from a CSV file
df = pd.read_csv('FuelConsumption.csv')  # Replace 'your_data.csv' with the actual file path

# Separate the features (X) and the target variable (y)
X = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]  # Replace with your actual feature column names
y = df['CO2EMISSIONS']  # Replace with your actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set for both models
rf_predictions = rf.predict(X_test)

# Calculate MSE for both models
rf_mse = mean_squared_error(y_test, rf_predictions)

# Calculate MAE for both models
rf_mae = mean_absolute_error(y_test, rf_predictions)

# Calculate R-squared for both models
rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest Regressor:")
print("Mean Squared Error (MSE):", rf_mse)
print("Mean Absolute Error (MAE):", rf_mae)
print("R-squared:", rf_r2)
