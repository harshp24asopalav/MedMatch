import os
from dev.models.train import train_model
from dev.models.evaluate import evaluate_model

def train_model_function(data_id, model_type):
    
    data_path = os.path.join('data-collection', 'raw-data', f'{data_id}.csv')
    model_path = os.path.join('trained-models')
    
    # Train the model
    result = train_model(data_path, model_path)

    return result

def evaluate_model_function(model_id, validation_data_id):
    
    # Load the model and validation data
    model_path = os.path.join('trained-models', f'{model_id}.pkl')
    validation_data_path = os.path.join('data-collection', 'clean-data', f'{validation_data_id}.csv')

    # Evaluate the model
    result = evaluate_model(model_path, validation_data_path)

    return result