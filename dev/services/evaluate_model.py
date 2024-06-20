from flask import request, jsonify
from dev.utils.model_management import evaluate_model_function

def evaluate_model():
    model_id = request.json.get('modelId')
    validation_data_id = request.json.get('validationDataId')
    
    if not model_id or not validation_data_id:
        return jsonify({'error': 'Missing parameters'}), 400
    
    result = evaluate_model_function(model_id, validation_data_id)
    
    return jsonify(result), 200