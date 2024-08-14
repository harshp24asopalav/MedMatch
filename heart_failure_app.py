from flask import Flask, request, jsonify
from heart_failure import HeartFailure
import logging
import os

app = Flask(__name__)
hf = HeartFailure(use_pretrained=False)

# Configure logging to save logs to a file
log_file_path = os.path.join('logs', 'hf_logs.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'  # Append to the log file
)

@app.route('/hf/load_data', methods=['POST'])
def load_data():
    logging.info("API call: /load_data")
    hf.load_data()
    return jsonify({"status": "Data loaded successfully"})

@app.route('/hf/process_data', methods=['POST'])
def process_data():
    logging.info("API call: /process_data")
    if hf.df is None:
        return jsonify({"error": "Data not loaded. Please call /load_data first."})
    hf.process_data()
    return jsonify({"status": "Data processed successfully"})

@app.route('/hf/model', methods=['POST'])
def build_model():
    logging.info("API call: /build_model")
    hf.model()
    return jsonify({"status": "Model built successfully"})

@app.route('/hf/train_model', methods=['POST'])
def train_model():
    logging.info("API call: /train_model")
    if hf.X_train is None or hf.y_train is None:
        return jsonify({"error": "Data not processed. Please call /process_data first."})
    hf.train_model()
    return jsonify({"status": "Model trained successfully"})

@app.route('/hf/save_model', methods=['POST'])
def save_model():
    logging.info("API call: /save_model")
    if hf.rf_model is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    hf.save_model()
    return jsonify({"status": "Model saved successfully"})

@app.route('/hf/evaluate', methods=['POST'])
def evaluate_model():
    logging.info("API call: /evaluate")
    if hf.rf_model is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    hf.evaluate()
    return jsonify({"status": "Model evaluated successfully"})

if __name__ == '__main__':
    app.run(debug=True)