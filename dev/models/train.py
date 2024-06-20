import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_model(data_path, model_path):
    # Load the data
    df = pd.read_csv(data_path)

    # Split the data into features and target
    X = df.drop(['DEATH_EVENT'], axis=1)
    y = df['DEATH_EVENT']

    # Scale the data
    h_scaler = StandardScaler()
    X_scaled = h_scaler.fit_transform(X)

    # Save the scaler
    scaler_filename = 'scaler_h.pkl'
    with open(os.path.join(model_path, scaler_filename), 'wb') as file:
        pickle.dump(h_scaler, file)

    # Split the data into train and test sets
    X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=44, shuffle=True)

    # Define and train the model
    Rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, criterion='gini', random_state=0, n_jobs=-1)
    Rf_model.fit(X_scaled_train, y_train.ravel())

    # Save the model
    model_filename = 'random_forest_model.pkl'
    with open(os.path.join(model_path, model_filename), 'wb') as file:
        pickle.dump(Rf_model, file)

    return {'message': 'Model training complete', 'model_path': os.path.join(model_path, model_filename)}
