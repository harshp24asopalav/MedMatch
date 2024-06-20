import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_model(clean_data_id):
    clean_data_path = os.path.join('data-collection', 'clean-data', f'{clean_data_id}_clean.csv')
    model_path = os.path.join('trained-model', f'{clean_data_id}_model.pkl')

    # Load the cleaned data
    try:
        df = pd.read_csv(clean_data_id)
    except FileNotFoundError:
        return {'error':'Clean data file not found'}, 400

    # Split the data into features and target
    X = df.drop(['DEATH_EVENT'], axis=1)
    y = df['DEATH_EVENT']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44, shuffle=True)

    # Define and train the model
    Rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, criterion='gini', random_state=0, n_jobs=-1)
    Rf_model.fit(X_train, y_train.ravel())

    # Save the model
    
    with open(model_path, 'wb') as file:
        pickle.dump(Rf_model, file)

    return {'message': 'Model training complete', 'model_path': os.path.join(model_path)}
