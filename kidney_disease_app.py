from flask import Flask, request, jsonify
from kidney_disease import KidneyDisease
import os, logging

app = Flask(__name__)
kd = KidneyDisease(use_pretrained=False)

# Configure logging to save logs to a file
log_file_path = os.path.join('logs', 'kd_logs.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'  # Append to the log file
)

@app.route('/kd/load_data', methods=['POST'])
def load_data():
    logging.info("API call: /load_data")
    kd.load_data()
    return jsonify({"status": "Data loaded successfully"})

@app.route('/kd/process_data', methods=['POST'])
def process_data():
    logging.info("API call: /process_data")
    if kd.df is None:
        return jsonify({"error": "Data not loaded. Please call /load_data first."})
    kd.process_data()
    return jsonify({"status": "Data processed successfully"})

@app.route('/kd/model', methods=['POST'])
def build_model():
    logging.info("API call: /model")
    kd.model()
    return jsonify({"status": "Model built successfully"})

@app.route('/kd/train_model', methods=['POST'])
def train_model():
    logging.info("API call: /train_model")
    if kd.X_train is None or kd.y_train is None:
        return jsonify({"error": "Data not processed. Please call /process_data first."})
    kd.train_model()
    return jsonify({"status": "Model trained successfully"})

@app.route('/kd/save_model', methods=['POST'])
def save_model():
    logging.info("API call: /save_model")
    if kd.rf_classifier is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    kd.save_model()
    return jsonify({"status": "Model saved successfully"})

@app.route('/kd/evaluate', methods=['POST'])
def evaluate_model():
    logging.info("API call: /evaluate")
    if kd.rf_classifier is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    kd.evaluate()
    return jsonify({"status": "Model evaluated successfully"})

if __name__ == '__main__':
    app.run(debug=True)
