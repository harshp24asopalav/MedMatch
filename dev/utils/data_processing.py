# # dev/utils/data_processing.py
# import os
# import pandas as pd

# def preprocess_data_function(raw_data_id):
#     raw_data_path = os.path.join('data-collection', 'raw-data', f'{raw_data_id}.csv')
#     clean_data_path = os.path.join('data-collection', 'clean-data', f'{raw_data_id}_clean.csv')
    
#     # Load raw data
#     try:
#         df = pd.read_csv(raw_data_path)
#     except FileNotFoundError:
#         return {'error': 'Raw data file not found'}, 400
    
#     # Example preprocessing steps
#     df.dropna(inplace=True)  # Remove rows with null valuesz
#     # Add more preprocessing steps as needed (e.g., standardizing formats)


    
#     # Save cleaned data
#     df.to_csv(clean_data_path, index=False)
    
#     return {'message': 'Data preprocessed successfully', 'cleanDataPath': clean_data_path}, 200


# dev/utils/data_processing.py
# dev/utils/data_processing.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data_function(raw_data_id):
    raw_data_path = os.path.join('data-collection', 'raw-data', f'{raw_data_id}.csv')
    clean_data_path = os.path.join('data-collection', 'clean-data', f'{raw_data_id}_clean.csv')
    scaler_path = os.path.join('data-collection', 'clean-data', f'{raw_data_id}_scaler.pkl')
    
    # Load raw data
    try:
        df = pd.read_csv(raw_data_path)
    #  return{'success': 'Raw data file found'},200
    except FileNotFoundError:
        return {'error': 'Raw data file not found'}, 400
    
    
    df.dropna(inplace=True)  # Remove rows with null values
    X = df.drop(['DEATH_EVENT'], axis=1)
    y = df['DEATH_EVENT']

    # Scale the data
    h_scaler = StandardScaler()
    X_scaled = h_scaler.fit_transform(X)

    # Save the scaler to a file
    with open(scaler_path, 'wb') as file:
        pickle.dump(h_scaler, file)

    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=44, shuffle=True)

    # Save cleaned data
    df_clean = pd.DataFrame(X_scaled_train, columns=X.columns)
    df_clean['DEATH_EVENT'] = y_train.values
    df_clean.to_csv(clean_data_path, index=False)
    
    return {'message': 'Data preprocessed successfully', 'cleanDataPath': clean_data_path, 'scalerPath': scaler_path}, 200
