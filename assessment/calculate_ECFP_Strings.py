# this computes the SMILES string into ECFP fingerprints. 
# It then stores the ECFP in a new table in the assay_database.db 

import sqlite3
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
import numpy as np

def compute_ecfp4_fingerprints(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)

def fingerprint_to_binary(fp):
    # Convert the fingerprint to a binary string
    return ''.join([str(b) for b in fp])

def compute_and_store_fingerprints(database_file):
    # Connect to the SQLite database to retrieve SMILES
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    smiles_data = pd.read_sql_query("SELECT CID, SMILES FROM rule_of_five_compliant_compounds", conn)

    # Compute ECFP4 fingerprints
    smiles_data['ECFP4_Fingerprint'] = smiles_data['SMILES'].apply(compute_ecfp4_fingerprints)

    # Create a new table for fingerprints
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ecfp4_fingerprints (
        CID TEXT PRIMARY KEY,
        ECFP4_Fingerprint TEXT
    )
    ''')

    # Insert fingerprints into the new table
    for _, row in smiles_data.iterrows():
        binary_fp = fingerprint_to_binary(row['ECFP4_Fingerprint'])
        cursor.execute('''
        INSERT OR REPLACE INTO ecfp4_fingerprints (CID, ECFP4_Fingerprint)
        VALUES (?, ?)
        ''', (row['CID'], binary_fp))

    # Commit changes to the database
    conn.commit()
    conn.close()

if __name__ == "__main__":
    database_file = 'assays_database.db'
    compute_and_store_fingerprints(database_file)
