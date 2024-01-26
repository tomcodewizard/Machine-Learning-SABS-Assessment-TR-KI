import sqlite3
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Helper function to compute descriptors for a single molecule
def compute_descriptors(molecule):
    descriptors = {d[0]: d[1](molecule) for d in Descriptors.descList}
    return pd.Series(descriptors)

def compute_and_store_descriptors(database_file):
    # Connect to the SQLite database to retrieve SMILES
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    smiles_data = pd.read_sql_query("SELECT CID, SMILES FROM rule_of_five_compliant_compounds", conn)

    # List to store descriptor data
    all_descriptors = []

    for _, row in smiles_data.iterrows():
        molecule = Chem.MolFromSmiles(row['SMILES'])
        if molecule is not None:
            descriptors = compute_descriptors(molecule)
            all_descriptors.append(descriptors)
        else:
            # Append None values if the molecule is None
            all_descriptors.append(pd.Series([None]*len(Descriptors.descList)))

    # Convert the list of Series to a DataFrame
    descriptors_df = pd.DataFrame(all_descriptors)

    # Combine CID and Descriptors into one DataFrame
    combined_data = pd.concat([smiles_data['CID'], descriptors_df], axis=1)

    # Create a new table for molecular descriptors
    combined_data.to_sql('molecular_descriptors', conn, if_exists='replace', index=False)

    # Commit changes to the database and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    database_file = 'assays_database.db'
    compute_and_store_descriptors(database_file)

