# dev/app.py
from flask import Flask, request, jsonify
from services.upload_training_data import upload_training_data
from services.train_model import train_model
from services.evaluate_model import evaluate_model

app = Flask(__name__)

app.add_url_rule('/upload-training-data', 'upload_training_data', upload_training_data, methods=['POST'])
app.add_url_rule('/train-model', 'train_model', train_model, methods=['POST'])
app.add_url_rule('/evaluate-model', 'evaluate_model', evaluate_model, methods=['POST'])

if __name__ == '__main__':
    port = 54678
    app.run(debug=True, port=port)
