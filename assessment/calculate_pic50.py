# this retrives IC50 data from the assays table in the database and calculates pIC50 values 
# It then plots the pIC50 values and their frequency

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_pic50(ic50):
    return -np.log10(ic50 * 1e-6)  # IC50 values are assumed to be in uM

def compute_and_plot_pic50(database_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Retrieve assay data for rule_of_five_compliant_compounds
    query = '''
    SELECT a.CID, a.f_avg_IC50, a.r_avg_IC50
    FROM assays a
    INNER JOIN rule_of_five_compliant_compounds c ON a.CID = c.CID
    '''
    assay_data = pd.read_sql_query(query, conn)
    
    # Compute pIC50 values
    assay_data['f_avg_pIC50'] = assay_data['f_avg_IC50'].apply(calculate_pic50)
    assay_data['r_avg_pIC50'] = assay_data['r_avg_IC50'].apply(calculate_pic50)

    # Create a new table for pIC50 values
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pIC50_VALUES (
        CID TEXT PRIMARY KEY,
        f_avg_pIC50 REAL,
        r_avg_pIC50 REAL
    )
    ''')

    # Insert pIC50 values into the new table
    for _, row in assay_data.iterrows():
        cursor.execute('''
        INSERT OR REPLACE INTO pIC50_values (CID, f_avg_pIC50, r_avg_pIC50)
        VALUES (?, ?, ?)
        ''', (row['CID'], row['f_avg_pIC50'], row['r_avg_pIC50']))

    # Commit chnages to the database
    conn.commit()

    # Plotting the distribution of pIC50 values
    plt.figure(figsize=(10, 6))
    #plt.hist(assay_data['f_avg_pIC50'].dropna(), bins=30, alpha=0.5, label='f_avg_pIC50')
    #plt.hist(assay_data['r_avg_pIC50'].dropna(), bins=30, alpha=0.5, label='r_avg_pIC50')

    n_bins = 30
    bin_edges = np.linspace(min(assay_data[['f_avg_pIC50', 'r_avg_pIC50']].min()), max(assay_data[['f_avg_pIC50', 'r_avg_pIC50']].max()), n_bins + 1)

    plt.hist(assay_data['f_avg_pIC50'].dropna(), bins=bin_edges, alpha=0.5, label='f_avg_pIC50', density=True, edgecolor='black')
    plt.hist(assay_data['r_avg_pIC50'].dropna(), bins=bin_edges, alpha=0.5, label='r_avg_pIC50', density=True, edgecolor='black')

    plt.xlabel('pIC50 (pIC50 units)')
    plt.ylabel('Frequency')
    plt.title('Distribution of pIC50 Values')
    plt.legend()
    plt.show()
    
    # Close the databse connection
    conn.close()

if __name__ == "__main__":
    database_file = 'assays_database.db'
    compute_and_plot_pic50(database_file)

