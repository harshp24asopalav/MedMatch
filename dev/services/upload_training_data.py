from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/upload-training-data', methods=['POST'])
def upload_training_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    description = request.form.get('description')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not description:
        return jsonify({'error': 'No description provided'}), 400

    data_dir = os.path.join(os.path.abspath(os.sep), "..", "data-collection", "raw-data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    save_path = os.path.join(data_dir, file.filename)
    file.save(save_path)

    return jsonify({'message': 'File uploaded successfully', 'description': description}), 200

