# Import libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.filterwarnings("ignore")
import os, logging

# Configure logging to save logs to a file
log_file_path = os.path.join('logs', 'cp_logs.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'  # Append to the log file
)

class CaloriePrediction:
    def __init__(self, use_pretrained=True) -> None:
        if use_pretrained:
            print("Loading model...")
            self.load_model()
        else:
            print("Training new model...")
            self.load_data()
            self.preprocess_data()
            self.split_data()
            self.train_model()
            self.save_model()
            self.evaluate_model()

    def load_data(self):
        # Load the dataset
        self.df = pd.read_csv("./data-collection/diet/diet.csv")

    def preprocess_data(self):
        # Preprocessing steps
        self.df = self.df.drop(columns=['Unnamed: 0', 'BMI_tags', 'Label'])
        self.df['gender'] = self.df['gender'].replace({'F': 0, 'M': 1})

    def split_data(self):
        # Separate features and target variable and split the data
        X = self.df.drop(columns=['calories_to_maintain_weight'])
        y = self.df['calories_to_maintain_weight']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        # Train the RandomForest model
        self.rf_model = RandomForestRegressor(random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)

    def save_model(self):
        # Save the trained RandomForest model
        filename = './trained-model/cal_version_1.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.rf_model, file)
        logging.info("Model saved successfully.")
    
    def load_model(self):
        # Load the RandomForest model
        filename = './trained-model/cal_version_1.pkl'
        with open(filename, 'rb') as file:
            self.rf_model = pickle.load(file)
        logging.info("Model loaded successfully.")

    def evaluate_model(self):
        # Make predictions and evaluate the model
        y_pred = self.rf_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse_rf = np.sqrt(mse)
        print("RMSE for Random Forest Regression:", rmse_rf)

    def predict(self, features):
        # feature_name = ['age','weight(kg)', 'height(m)', 'gender', 'BMI', 'BMR', 'activity_level']
        features_df = pd.DataFrame([features])
        prediction = self.rf_model.predict(features_df)
        return prediction

# Example usage
# calorie_model = CaloriePrediction(use_pretrained=False)  # Set to True to load an existing model
# Create an instance of the CaloriePrediction class using the pretrained model
"""cp = CaloriePrediction(use_pretrained=True)

# Prepare a sample input with the expected features
sample_input = {
    'age': 30,
    'weight(kg)': 70,
    'height(m)': 1.60,
    'gender': 1,  # Assuming '1' represents 'M' as per your encoding
    'BMI': 22.9,
    'BMR': 1650,
    'activity_level': 1.3  # Ensure the format matches your model's training data
}

# Predict using the sample input
predicted_value = cp.predict(sample_input)

# Print the predicted output
print("Predicted calorie maintenance level:", predicted_value)"""