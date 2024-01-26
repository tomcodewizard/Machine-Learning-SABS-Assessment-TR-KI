import re
import csv
import sqlite3
import pandas as pd
from pathlib import Path

# Load the CSV file
csv_file = 'covid_submissions_all_info.csv'
data = pd.read_csv(csv_file)

# Define a function to count the number of 'rule of five' violations
def rule_of_five_violations(row):
    violations = sum([
        row['HBD'] > 5,
        row['HBA'] > 10,
        row['MW'] >= 500,
        row['cLogP'] >= 5
    ])
    return violations

# Apply the function to the data
data['rule_of_five_violations'] = data.apply(rule_of_five_violations, axis=1)

# Filter out compounds that violate more than one rule
rule_of_five_data = data[data['rule_of_five_violations'] <= 1]

# Extract data for the 'compounds' and 'rule_of_five_compliant_compounds' tables
smiles_data = data[['CID', 'SMILES', 'HBD', 'HBA', 'MW', 'cLogP']].drop_duplicates()

# Select relevant columns for 'assays'
columns_of_interest = ['CID', 'f_avg_IC50', 'r_avg_IC50']
assay_data = data[columns_of_interest]

# Create a new SQLite database
database_file = 'assays_database.db'
conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# Create the 'compounds' table 
cursor.execute('''
    CREATE TABLE IF NOT EXISTS compounds (
        CID TEXT PRIMARY KEY,
        SMILES TEXT,
        HBD INTEGER,
        HBA INTEGER,
        MW REAL,
        cLogP REAL
    );           
''')

# Create the 'rule_of_five_compliant_compounds' table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS rule_of_five_compliant_compounds (
        CID TEXT PRIMARY KEY,
        SMILES TEXT,
        HBD INTEGER,
        HBA INTEGER,
        MW REAL,
        cLogP REAL
    );
''')

# Populate the 'compounds' table
for _, row in smiles_data.iterrows():
    cursor.execute('''
        INSERT OR IGNORE INTO compounds (CID, SMILES, HBD, HBA, MW, cLogP) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (row['CID'], row['SMILES'], row['HBD'], row['HBA'], row['MW'], row['cLogP']))

# Populate the 'rule_of_five_compliant_compounds' table
for _, row in rule_of_five_data.iterrows():
    cursor.execute('''
        INSERT OR IGNORE INTO rule_of_five_compliant_compounds (CID, SMILES, HBD, HBA, MW, cLogP) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (row['CID'], row['SMILES'], row['HBD'], row['HBA'], row['MW'], row['cLogP']))

# Create the 'assays' table with a foreign key
cursor.execute('''
    CREATE TABLE IF NOT EXISTS assays (
        CID TEXT,
        f_avg_IC50 REAL,
        r_avg_IC50 REAL,
        FOREIGN KEY (CID) REFERENCES compounds(CID)
    );
''')

# Populate the 'assays' data into the table
for _, row in assay_data.iterrows():
    cursor.execute('''
        INSERT INTO assays (CID, f_avg_IC50, r_avg_IC50) 
        VALUES (?, ?, ?)
    ''', (row['CID'], row['f_avg_IC50'], row['r_avg_IC50']))

# Commit changes and close the connection
conn.commit()

# Demonstrate an SQL query for joining the two tables
cursor.execute('''
    SELECT c.CID, c.SMILES, a.f_avg_IC50, a.r_avg_IC50
    FROM compounds c
    JOIN assays a ON c.CID = a.CID
''')

# Fetch and print the result of the query
joined_data = cursor.fetchall()
for row in joined_data:
    print(row)

conn.close()

print("Database and tables created successfully with the data.")