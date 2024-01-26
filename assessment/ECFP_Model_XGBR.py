import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to convert binary strings back to binary vectors
def binary_to_vector(binary_string):
    return np.array([int(bit) for bit in binary_string])

# Connect to the SQLite database
conn = sqlite3.connect('assays_database.db')

# Retrieve ECFP fingerprints and pIC50 values
query = """
SELECT f.CID, f.ECFP4_Fingerprint, p.f_avg_pIC50
FROM ecfp4_fingerprints f
JOIN pIC50_values p ON f.CID = p.CID
"""
data = pd.read_sql_query(query, conn)
conn.close()

# Drop rows with NaN values in the pIC50 column
data = data.dropna(subset=['f_avg_pIC50'])

# Convert binary strings to vectors
data['ECFP4_Fingerprint'] = data['ECFP4_Fingerprint'].apply(binary_to_vector)

# Prepare the features and target variable
X = np.stack(data['ECFP4_Fingerprint'].values)
y = data['f_avg_pIC50'].values

# Split the data into training, validation, and test sets (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on the test set and calculate RMSE
y_pred_test = model.predict(X_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

print(f"Test RMSE: {rmse_test}")

# Predict on the validation set and calculate RMSE
y_pred_val = model.predict(X_val)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

print(f"Validation RMSE: {rmse_val}")

# Visualization
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.title('Actual vs. Predicted pIC50 Values - Test Set')
plt.xlabel('Actual pIC50 (pIC50 units)')
plt.ylabel('Predicted pIC50')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

residuals = y_test - y_pred_test
plt.scatter(y_test, residuals, alpha=0.5)
plt.title('Residuals vs. Actual pIC50 Values')
plt.xlabel('Actual pIC50 (pIC50 units)')
plt.ylabel('Residuals')
plt.axhline(y=0, color='k', linestyle='--')
plt.show()

plt.hist(residuals, bins=20, alpha=0.7, color='g', edgecolor='black')
plt.title('Distribution of Prediction Errors (Residuals)')
plt.xlabel('Prediction Error (Residual)')
plt.ylabel('Frequency')
plt.show()