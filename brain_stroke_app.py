from flask import Flask, request, jsonify
from brain_stroke import BrainStroke
import logging, os

app = Flask(__name__)
bs = BrainStroke(use_pretrained=False)

# Configure logging to save logs to a file
log_file_path = os.path.join('logs', 'bs_logs.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'  # Append to the log file
)

@app.route('/bs/load_data', methods=['POST'])
def load_data():
    logging.info("API call: /load_data")
    bs.load_data()
    return jsonify({"status": "Data loaded successfully"})

@app.route('/bs/process_data', methods=['POST'])
def process_data():
    logging.info("API call: /process_data")
    if bs.df is None:
        return jsonify({"error": "Data not loaded. Please call /load_data first."})
    bs.process_data()
    return jsonify({"status": "Data processed successfully"})

@app.route('/bs/model', methods=['POST'])
def build_model():
    logging.info("API call: /model")
    bs.model()
    return jsonify({"status": "Model built successfully"})

@app.route('/bs/train_model', methods=['POST'])
def train_model():
    logging.info("API call: /train_model")
    if bs.X_train is None or bs.y_train is None:
        return jsonify({"error": "Data not processed. Please call /process_data first."})
    bs.train_model()
    return jsonify({"status": "Model trained successfully"})

@app.route('/bs/save_model', methods=['POST'])
def save_model():
    logging.info("API call: /save_model")
    if bs.etc_model is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    bs.save_model()
    return jsonify({"status": "Model saved successfully"})

@app.route('/bs/evaluate', methods=['POST'])
def evaluate_model():
    logging.info("API call: /evaluate")
    if bs.etc_model is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    bs.evaluate()
    return jsonify({"status": "Model evaluated successfully"})

if __name__ == '__main__':
    app.run(debug=True)
