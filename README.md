- Go into assessment

- Before starting, unzip the assay_database file and make sure the name is actually 'assay_database.db'. 

- Run database.py

- This gives the assay_database.db database. (which can be visualused on SQLite)

- Run calculate_pic50.py to give pic50s (which is inseretd as a new table in assay_database)

- Run lipinski_XGBR.py or lipinski_RandomF.py or lipinski_SVMs.py to get models that predcit pIC50 values based on lipinski RO5

- This gives Test RMSE and Validation RMSE alongside other analysis.

- Run calculate_ECFP_Strings.py to get ECFP fingerprints (this is stored in the assay_database.py)

- Run ECFP_Model_XGBR.py or ECFP_Model_RandomF.py or ECFP_Model_SVMs.py to get models that predict pIC50 values based on ECFP fingerprints



- To get model that predicts pIC50 based on 200+ molecular properties (not just lipinski rule) go into new_analysis.

- Before starting, unzip the assay_database file and make sure the name is actually 'assay_database.db'.

- Run rdkit_computations.py

- This computes the descriptors based on the SMILES strings in the database and stores it as new table with 200+ columns (descriptors)

- Run descriptors_XGBR.py or descriptors_RF.py or descriptors_SVMs.py to get models that predict pIC50 values based on descriptors fingerprints.
