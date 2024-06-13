import os
import pandas as pd
from flask import request, jsonify
from dev.utils.data_processing import preprocess_data_function

def preprocess_data():
    raw_data_id = request.json.get('rawDataId')
    if not raw_data_id:
        return jsonify({'error': 'Missing parameters'}), 400
    
    result = preprocess_data_function(raw_data_id)
    
    return jsonify(result), 200