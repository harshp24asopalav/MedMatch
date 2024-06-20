import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_model(model_path, data_path):
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Load the data
    df = pd.read_csv(data_path)

    # Split the data into features and target
    X = df.drop(['DEATH_EVENT'], axis=1)
    y = df['DEATH_EVENT']

    # Load the scaler and scale the data
    scaler_path = os.path.join(os.path.dirname(model_path), 'scaler_h.pkl')
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    X_scaled = scaler.transform(X)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)

    return {'message': 'Model evaluation complete', 'accuracy': accuracy}
