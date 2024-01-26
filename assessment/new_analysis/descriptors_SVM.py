import sqlite3
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(database_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_file)

    # Load pIC50 values
    pic50_data = pd.read_sql_query("SELECT CID, f_avg_pIC50 FROM pIC50_VALUES", conn)

    # Load molecular descriptors
    molecular_descriptors = pd.read_sql_query("SELECT * FROM molecular_descriptors", conn)

    # Close the database connection
    conn.close()

    # Merge datasets on CID
    merged_data = pd.merge(pic50_data, molecular_descriptors, on='CID')

    # Handle null values
    merged_data = merged_data.dropna()

    return merged_data

def train_and_evaluate(data):
    # Separate features and target variable
    X = data.drop(['CID', 'f_avg_pIC50'], axis=1)
    y = data['f_avg_pIC50']

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Here we initialise the SVM regressor
    model = SVR(kernel='rbf') #, C=1.0, epsilon=0.1, C=1.0, epsilon=0.1)
   

    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Predict on test and validation sets
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)

    # Calculate RMSE for test and validation sets
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    print(f"Test RMSE: {rmse_test}")
    print(f"Validation RMSE: {rmse_val}")

    # Plotting
    plt.figure(figsize=(12, 6))

    # Scatter plot of Actual vs. Predicted pIC50 values
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual pIC50 (pIC50 units)')
    plt.ylabel('Predicted pIC50 (pIC50 units)')
    plt.title('Actual vs. Predicted pIC50 Values')

    # Histogram of residuals (Prediction Errors)
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred_test
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='g')
    plt.xlabel('Prediction Error (Residuals)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Residuals)')

    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":
    database_file = 'assays_database.db'
    data = load_data(database_file)
    model = train_and_evaluate(data)
