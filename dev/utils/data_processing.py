# dev/utils/data_processing.py
import os
import pandas as pd

def preprocess_data_function(raw_data_id):
    raw_data_path = os.path.join('data-collection', 'raw-data', f'{raw_data_id}.csv')
    clean_data_path = os.path.join('data-collection', 'clean-data', f'{raw_data_id}_clean.csv')
    
    # Load raw data
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        return {'error': 'Raw data file not found'}, 400
    
    # Example preprocessing steps
    df.dropna(inplace=True)  # Remove rows with null values
    # Add more preprocessing steps as needed (e.g., standardizing formats)
    
    # Save cleaned data
    df.to_csv(clean_data_path, index=False)
    
    return {'message': 'Data preprocessed successfully', 'cleanDataPath': clean_data_path}, 200