# this creates a model that predicts pIC50 values trianed on Molecualr descriptors (lipinski rule of 5)
# Using RandomForest

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Connect to the SQLite database
conn = sqlite3.connect('assays_database.db')

# Retrieve data from the database
query = """
SELECT c.HBD, c.HBA, c.MW, c.cLogP, p.f_avg_pIC50
FROM rule_of_five_compliant_compounds c
JOIN pIC50_values p ON c.CID = p.CID
"""
data = pd.read_sql_query(query, conn)
conn.close()

# Drop rows with NaN values in any column
data.dropna(inplace=True)

# Define the features and target
features = ['HBD', 'HBA', 'MW', 'cLogP']
X = data[features]
y = data['f_avg_pIC50']

# Split the data into training, validation, and test sets (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on the test set and calculate RMSE
y_pred_test = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Test RMSE: {rmse_test}")

# Predict on the validation set and calculate RMSE
y_pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {rmse_val}")

# Optionally: Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
print("Feature importances:", feature_importance_dict)

# Scatter plot of Actual vs. Predicted pIC50 Values
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.title('Actual vs. Predicted pIC50 Values')
plt.xlabel('Actual pIC50 (pIC50 units)')
plt.ylabel('Predicted pIC50 (pIC50 units)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line for reference
plt.show()

# Residuals plot
residuals = y_test - y_pred_test
plt.hist(residuals, bins=30, alpha=0.7, color='g', edgecolor='black')
plt.title('Distribution of Prediction Errors (Residuals)')
plt.xlabel('Prediction Error (Residual)')
plt.ylabel('Frequency')
plt.show()


