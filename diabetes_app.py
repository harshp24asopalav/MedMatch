from flask import Flask, request, jsonify
from diabetes import Diabetes
import logging, os

app = Flask(__name__)
di = Diabetes(use_pretrained=False)

# Configure logging to save logs to a file
log_file_path = os.path.join('logs', 'di_logs.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'  # Append to the log file
)

@app.route('/di/load_data', methods=['POST'])
def load_data():
    logging.info("API call: /load_data")
    di.load_data()
    return jsonify({"status": "Data loaded successfully"})

@app.route('/di/process_data', methods=['POST'])
def process_data():
    logging.info("API call: /process_data")
    if di.df is None:
        return jsonify({"error": "Data not loaded. Please call /load_data first."})
    di.process_data()
    return jsonify({"status": "Data processed successfully"})

@app.route('/di/model', methods=['POST'])
def build_model():
    logging.info("API call: /model")
    di.model()
    return jsonify({"status": "Model built successfully"})

@app.route('/di/train_model', methods=['POST'])
def train_model():
    logging.info("API call: /train_model")
    if di.X_res is None or di.y_res is None:
        return jsonify({"error": "Data not processed. Please call /process_data first."})
    di.train_model()
    return jsonify({"status": "Model trained successfully"})

@app.route('/di/save_model', methods=['POST'])
def save_model():
    logging.info("API call: /save_model")
    if di.xgb_model is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    di.save_model()
    return jsonify({"status": "Model saved successfully"})

@app.route('/di/evaluate', methods=['POST'])
def evaluate_model():
    logging.info("API call: /evaluate")
    if di.xgb_model is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    di.evaluate()
    return jsonify({"status": "Model evaluated successfully"})

if __name__ == '__main__':
    app.run(debug=True)
