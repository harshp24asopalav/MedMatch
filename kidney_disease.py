import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pickle
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv()

class KidneyDisease:
    def __init__(self, use_pretrained = True) -> None:

        self.kd_model_path = os.getenv('KD_MODEL', './trained-model/kd_version_1.pkl')
        self.kd_test_size = float(os.getenv('KD_TEST_SIZE', '0.2'))
        self.kd_random_state = int(os.getenv('KD_RANDOM_STATE', '42'))
        self.kd_n_estimators = int(os.getenv('KD_N_ESTIMATORS', '100'))

        if use_pretrained:
            print("Loading model...")
            #self.model = keras.models.load_model("./trained-model/kd_version_1.pkl")
            self.load_model()
        else:
            print("Training model...")
            self.load_data()
            self.process_data()
            self.model()
            self.train_model()
            self.save_model()
            self.evaluate()

    def load_data(self):
        df = pd.read_csv('./data-collection/Kidney_Disease/kidney.csv')
        self.df = df

    def process_data(self):
        X = self.df.drop(columns=['Class'])
        y = self.df['Class']

        """# Standardize the features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)"""
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.kd_test_size, random_state=self.kd_random_state)

    def model(self):
        rf_classifier = RandomForestClassifier(n_estimators=self.kd_n_estimators, random_state=42)
        self.rf_classifier = rf_classifier

    def train_model(self):
        self.rf_classifier.fit(self.X_train, self.y_train)

    def save_model(self):
        filename = './trained-model/kd_version_1.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.rf_classifier, file)

    def load_model(self):
        filename = self.kd_model_path
        with open(filename, 'rb') as file:
            self.rf_classifier = pickle.load(file)

    def evaluate(self):
        # Make predictions on the test set
        y_kd_pred = self.rf_classifier.predict(self.X_test)
        # Calculate the accuracy
        accuracy = accuracy_score(self.y_test, y_kd_pred)
        # Print the results
        print(f'Accuracy: {accuracy}')

    def predict(self, features):
        #feature_names = ['Bp', 'Sg', 'Al', 'Su', 'Rbc', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Wbcc', 'Rbcc', 'Htn']
        features_df = pd.DataFrame([features])
        # scaled_features = self.scaler.transform(features_df)
        prediction = self.rf_classifier.predict(features_df)
        return bool(prediction[0])

# Test Model
"""kd = KidneyDisease(use_pretrained=True)
input_data = {
        "Bp":80.0,
        "Sg":1.020,
        "Al":1.0,
        "Su":0.0,
        "Rbc":1.0,
        "Bu":36.0,
        "Sc":1.2,
        "Sod":137.53,
        "Pot":4.63,
        "Hemo":15.4,
        "Wbcc": 7800.0,
        "Rbcc": 5.20,
        "Htn": 1.0
}

prediction = kd.predict(input_data)
print(f'Prediction: {bool(prediction)}')"""