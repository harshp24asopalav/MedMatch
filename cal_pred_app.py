from flask import Flask, request, jsonify
from Cal_Pred import CaloriePrediction
import logging, os

app = Flask(__name__)
cp = CaloriePrediction(use_pretrained=False)

# Configure logging to save logs to a file
log_file_path = os.path.join('logs', 'cp_logs.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'  # Append to the log file
)


@app.route('/cal/load_data', methods=['POST'])
def load_data():
    logging.info("API call: /load_data")
    cp.load_data()
    return jsonify({"status": "Data loaded successfully"})

@app.route('/cal/preprocess_data', methods=['POST'])
def preprocess_data():
    logging.info("API call: /preprocess_data")
    if cp.df is None:
        return jsonify({"error": "Data not loaded. Please call /load_data first."})
    cp.preprocess_data()
    return jsonify({"status": "Data preprocessed successfully"})

@app.route('/cal/split_data', methods=['POST'])
def split_data():
    logging.info("API call: /split_data")
    if cp.df is None:
        return jsonify({"error": "Data not loaded. Please call /load_data first."})
    if cp.df.empty:
        return jsonify({"error": "Data not preprocessed. Please call /preprocess_data first."})
    cp.split_data()
    return jsonify({"status": "Data split into training and testing sets successfully"})

@app.route('/cal/train_model', methods=['POST'])
def train_model():
    logging.info("API call: /train_model")
    if cp.X_train is None or cp.y_train is None:
        return jsonify({"error": "Data not split. Please call /split_data first."})
    cp.train_model()
    return jsonify({"status": "Model trained successfully"})

@app.route('/cal/save_model', methods=['POST'])
def save_model():
    logging.info("API call: /save_model")
    if cp.rf_model is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    cp.save_model()
    return jsonify({"status": "Model saved successfully"})

@app.route('/cal/evaluate_model', methods=['POST'])
def evaluate_model():
    logging.info("API call: /evaluate_model")
    if cp.rf_model is None:
        return jsonify({"error": "Model not trained. Please call /train_model first."})
    cp.evaluate_model()
    return jsonify({"status": "Model evaluated successfully"})

if __name__ == '__main__':
    app.run(debug=True)
