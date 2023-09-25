import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data from a CSV file
df = pd.read_csv('FuelConsumption.csv')  # Replace 'your_data.csv' with the actual file path

# Separate the features (X) and the target variable (y)
X = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]  # Replace with your actual feature column names
y = df['CO2EMISSIONS']  # Replace with your actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree regressor model
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
tree_predictions = decision_tree.predict(X_test)

# Calculate MSE
tree_mse = mean_squared_error(y_test, tree_predictions)

# Calculate MAE
tree_mae = mean_absolute_error(y_test, tree_predictions)

# Calculate R-squared
tree_r2 = r2_score(y_test, tree_predictions)

print("Decision Tree Regressor:")
print("Mean Squared Error (MSE):", tree_mse)
print("Mean Absolute Error (MAE):", tree_mae)
print("R-squared:", tree_r2)