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

def compute_and_cluster(database_file):
    # Connect to the SQLite database to retrieve SMILES
    conn = sqlite3.connect(database_file)
    smiles_data = pd.read_sql_query("SELECT CID, SMILES FROM rule_of_five_compliant_compounds", conn)
    conn.close()

    # Compute ECFP fingerprints
    smiles_data['ECFP4_Fingerprint'] = smiles_data['SMILES'].apply(compute_ecfp4_fingerprints)

    # Convert fingerprints to a numpy array for clustering
    #fp_list = smiles_data['ECFP4_Fingerprint'].tolist()
    #fp_array = np.array([list(fp) for fp in fp_list])

    # Compute Tanimoto similarity matrix
    num_compounds = len(smiles_data)
    similarity_matrix = np.zeros((num_compounds, num_compounds))
    for i in range(num_compounds):
        for j in range(num_compounds):
            similarity_matrix[i, j] = DataStructs.FingerprintSimilarity(smiles_data['ECFP4_Fingerprint'][i],smiles_data['ECFP4_Fingerprint'][j])

    # Perform clustering (using Ward's method)
    Z = ward(1 - similarity_matrix)  # Ward's method expects distances, so we use 1 - similarity

    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=smiles_data['CID'].tolist())
    plt.title("Compound Clustering based on ECFP4 Tanimoto Similarity")
    plt.xlabel("Compound ID")
    plt.ylabel("Distance")
    plt.show()

if __name__ == "__main__":
    database_file = 'assays_database.db'
    compute_and_cluster(database_file)
