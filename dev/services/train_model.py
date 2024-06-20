from flask import request, jsonify
from dev.utils.model_management import train_model_function

def train_model():
    data_id = request.json.get('dataId')
    model_type = request.json.get('modelType')
    if not data_id or not model_type:
        return jsonify({'error': 'Missing parameters'}), 400
    
    result = train_model_function(data_id, model_type)
    
    return jsonify(result), 200