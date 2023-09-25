import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data from a CSV file
df = pd.read_csv('FuelConsumption.csv')  # Replace 'your_data.csv' with the actual file path

# Separate the features (X) and the target variable (y)
X = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]  # Replace with your actual feature column names
y = df['CO2EMISSIONS']  # Replace with your actual target column name


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Multi-layer Perceptron (MLP) Regressor model
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions on the test set
mlp_predictions = mlp.predict(X_test)

# Calculate MSE
mlp_mse = mean_squared_error(y_test, mlp_predictions)

# Calculate MAE
mlp_mae = mean_absolute_error(y_test, mlp_predictions)

# Calculate R-squared
mlp_r2 = r2_score(y_test, mlp_predictions)

print("Multi-layer Perceptron (MLP) Regressor:")
print("Mean Squared Error (MSE):", mlp_mse)
print("Mean Absolute Error (MAE):", mlp_mae)
print("R-squared:", mlp_r2)