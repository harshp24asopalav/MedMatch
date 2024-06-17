import os
from flask import request, jsonify

def upload_training_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    description = request.form.get('description')
    if not description:
        return jsonify({'error': 'No description provided'}), 400
    
    save_path = os.path.join('data-collection', 'raw-data', file.filename)
    file.save(save_path)
    
    # Add further processing here if needed
    
    return jsonify({'message': 'File uploaded successfully', 'description': description}), 200