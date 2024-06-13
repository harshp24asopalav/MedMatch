from flask import Flask, request, jsonify
from dev.services.upload_training_data import upload_training_data
from dev.services.train_model import train_model
from dev.services.evaluate_model import evaluate_model
from dev.services.preprocess_data import preprocess_data


app = Flask(__name__)

app.add_url_rule('/upload-training-data', 'upload_training_data', upload_training_data, methods=['POST'])
app.add_url_rule('/train-model', 'train_model', train_model, methods=['POST'])
app.add_url_rule('/evaluate-model', 'evaluate_model', evaluate_model, methods=['POST'])
app.add_url_rule('/preprocess-data', 'preprocess_data', preprocess_data, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
