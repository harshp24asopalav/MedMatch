from flask import request, jsonify
from dev.models.train import train_model

def train_model():
    clean_data_id = request.json.get('cleandataId')
    # model_type = request.json.get('modelType')
    if not clean_data_id:
        return jsonify({'error': 'Missing parameters'}), 400
    
    result = train_model(clean_data_id)
    
    return jsonify(result), 200
