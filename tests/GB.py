import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data from a CSV file
df = pd.read_csv('FuelConsumption.csv')  # Replace 'your_data.csv' with the actual file path

# Separate the features (X) and the target variable (y)
X = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]  # Replace with your actual feature column names
y = df['CO2EMISSIONS']  # Replace with your actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gradient Boosting Regressor model
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_regressor.fit(X_train, y_train)

# Make predictions on the test set
gb_predictions = gb_regressor.predict(X_test)

# Calculate MSE
gb_mse = mean_squared_error(y_test, gb_predictions)

# Calculate MAE
gb_mae = mean_absolute_error(y_test, gb_predictions)

# Calculate R-squared
gb_r2 = r2_score(y_test, gb_predictions)

print("Gradient Boosting Regressor:")
print("Mean Squared Error (MSE):", gb_mse)
print("Mean Absolute Error (MAE):", gb_mae)
print("R-squared:", gb_r2)
