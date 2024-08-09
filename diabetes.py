# Import libraries 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import warnings
import os, logging
warnings.filterwarnings("ignore")


# Configure logging to save logs to a file
log_file_path = os.path.join('logs', 'di_logs.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'  # Append to the log file
)

class Diabetes:
    def __init__(self, use_pretrained = True) -> None:
        if use_pretrained:
            print("Loading model...")
            self.load_model()
        else:
            print("Training new model...")
            self.load_data()
            self.process_data()
            self.model()
            self.train_model()
            self.save_model()
            self.evaluate()

    def load_data(self):
        df = pd.read_csv('./data-collection/Diabetes/diabetes_prediction_dataset.csv')
        self.df = df

    def process_data(self):
        self.df=self.df.drop(self.df[self.df["gender"]=="Other"].index)
        # Encode the data
        self.le = LabelEncoder()
        self.df['gender'] = self.le.fit_transform(self.df['gender'])
        self.df['smoking_history'] = self.le.fit_transform(self.df['smoking_history'])

        X = self.df.drop(columns=['diabetes'])
        y = self.df['diabetes']

        # Assuming X and y are your features and target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        self.X_res, self.y_res = smote.fit_resample(self.X_train, self.y_train)

    def model(self):
        xgb_model = XGBClassifier(class_weight='balanced', random_state=42)
        self.xgb_model = xgb_model

    def train_model(self):
        self.xgb_model.fit(self.X_res, self.y_res)

    def save_model(self):
        filename = './trained-model/di_version_1.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.xgb_model, file)
        logging.info("Model saved successfully.")

    def load_model(self):
        filename = './trained-model/di_version_1.pkl'
        with open(filename, 'rb') as file:
            self.xgb_model = pickle.load(file)
        logging.info("Model loaded successfully.")

    def evaluate(self):
        # Make predictions on the test set
        y_di_pred = self.xgb_model.predict(self.X_test)
        # Calculate the accuracy
        accuracy = accuracy_score(self.y_test, y_di_pred)
        # Print the results
        print(f'Accuracy: {accuracy}')

    def predict(self, features):
        # feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        features_df = pd.DataFrame([features], columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
    
        self.le_gender = LabelEncoder()
        self.le_smoking = LabelEncoder()
        # Use the previously fitted LabelEncoders
        features_df['gender'] = self.le_gender.fit_transform(features_df['gender'])
        features_df['smoking_history'] = self.le_smoking.fit_transform(features_df['smoking_history'])
        
        prediction = self.xgb_model.predict(features_df)
        return bool(prediction[0])

"""di = Diabetes(use_pretrained=False)
features = {
    "gender" : 'Male',
    "age": 50,
    "hypertension": 1,
    "heart_disease" : 0,
    "smoking_history": 'current',
    "bmi" : 27.32,
    "HbA1c_level" : 5.7,
    "blood_glucose_level" : 260
}
prediction = di.predict(features)
print(bool(prediction))"""



