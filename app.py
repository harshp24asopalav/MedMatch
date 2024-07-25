from flask import Flask, request, jsonify
import io  # For image handling
from tensorflow import keras
import sys
import os
import tempfile

# sys.path.insert(0, 'C:\\Users\\vishv\\OneDrive\\Documents\\AIML_2\\Project')
# print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
from brain_tumor import BrainTumor
from brain_stroke import BrainStroke
from diabetes import Diabetes
from heart_failure import HeartFailure
from kidney_disease import KidneyDisease
from lung_cancer import LungCancer
from QAChatbot import QAChatbot
from Cal_Pred import CaloriePrediction


bt = BrainTumor(use_pretrained=True)
bs = BrainStroke(use_pretrained=True)
di = Diabetes(use_pretrained=True)
hf = HeartFailure(use_pretrained=True)
kd = KidneyDisease(use_pretrained=True)
lc = LungCancer(use_pretrained=True)
cb = QAChatbot()
cp = CaloriePrediction(use_pretrained=True)

app = Flask(__name__)


@app.route('/api/is_tumor', methods=['POST'])
def handle_is_tumor():
    if request.method == 'POST':
        #print(request.headers)
        if 'image' not in request.files:
            print('image is not in request')
            return jsonify({'error': 'Missing image data'}), 400

        image_file = request.files['image']
        print(image_file)
        # Handle potential image errors (e.g., invalid format)  
        try:
            image_bytes = image_file.read()
            # Process image using your ML model 
            is_tumor_result = bt.predict(image_bytes)
            return jsonify({'is_tumor': is_tumor_result})
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400

    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/api/is_stroke', methods=['POST'])
def handle_is_stroke():
    if request.method == 'POST':
        if not request.json:
            return jsonify({'error': 'No JSON data received'}), 400

        input_data = request.json
        
        # Ensure input data matches the required format
        expected_features = [
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
            'Residence_type', 'avg_glucose_level', 'bmi', 'work_type_Govt_job', 
            'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 
            'work_type_children', 'smoking_status_Unknown', 'smoking_status_formerly smoked', 
            'smoking_status_never smoked', 'smoking_status_smokes'
        ]
        
        for feature in expected_features:
            if feature not in input_data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Call the predict function with the input data
        is_stroke = bs.predict(input_data)
        
        return jsonify({'is_stroke': is_stroke})

    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/api/is_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Get the input data from the request
        input_data = request.json

        # Define the expected features
        expected_features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

        # Validate input data
        if not all(feature in input_data for feature in expected_features):
            return jsonify({"error": "Invalid input, missing one or more features"}), 400

        # Prepare the features for prediction
        features = {feature: input_data[feature] for feature in expected_features}

        # Make prediction
        prediction = di.predict(features)

        # Return the prediction result
        return jsonify({"diabetes_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/is_heart', methods=['POST'])
def handle_is_heart():
    if request.method == 'POST':
        if not request.json:
            return jsonify({'error': 'No JSON data received'}), 400

        input_data = request.json
        
        # Ensure input data matches the required format
        expected_features = [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 
            'sex', 'smoking', 'time'
        ]
        
        for feature in expected_features:
            if feature not in input_data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Call the predict function with the input data
        is_heart = hf.predict(input_data)
        
        # Convert the prediction result to a standard Python int type
        is_heart = int(is_heart)
        
        return jsonify({'is_heart': bool(is_heart)})

    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/api/is_kidney', methods=['POST'])
def handle_is_kidney_disease():
    if request.method == 'POST':
        if not request.json:
            return jsonify({'error': 'No JSON data received'}), 400

        input_data = request.json
        
        # Ensure input data matches the required format
        expected_features = [
            'Bp', 'Sg', 'Al', 'Su', 'Rbc', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Wbcc', 'Rbcc', 'Htn'
        ]
        
        for feature in expected_features:
            if feature not in input_data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Call the predict function with the input data
        is_kidney_disease = kd.predict(input_data)
        
        return jsonify({'is_kidney_disease': is_kidney_disease})

    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/api/is_lung', methods=['POST'])
def predict_lung_cancer():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Use tempfile to create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            prediction = lc.predict(temp_file_path)
        finally:
            os.remove(temp_file_path)  # Remove the file after prediction
        
        return jsonify({'predicted_class': int(prediction[0])})


@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    if request.method == 'POST':
        data = request.get_json(force=True)  # Get JSON data from the request
        if 'question' not in data:
            return jsonify({'error': 'Missing question data'}), 400

        question = data['question']
        try:
            answers = cb.find_closest_answers(question)
            # Format the output to only include answers without scores
            formatted_answers = [answer for answer, _ in answers]
            return jsonify({'answers': formatted_answers})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Method not allowed'}), 405



@app.route('/api/calorie_maintenance', methods=['POST'])
def handle_calorie_maintenance():
    if request.method == 'POST':
        if not request.json:
            return jsonify({'error': 'No JSON data received'}), 400

        input_data = request.json
        
        # Ensure input data matches the required format
        expected_features = [
            'age', 'weight(kg)', 'height(m)', 'gender', 'BMI', 'BMR', 'activity_level'
        ]
        
        for feature in expected_features:
            if feature not in input_data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        
        # Call the predict function with the input data
        calorie_maintenance_level = cp.predict(input_data)
        
        return jsonify({'Calorie Maintenance Level': calorie_maintenance_level[0]})

    return jsonify({'error': 'Method not allowed'}), 405


if __name__ == '__main__':
    #import random  # For simulation purposes
    app.run(debug=True)
