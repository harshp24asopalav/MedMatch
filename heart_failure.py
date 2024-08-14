import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os, logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from dotenv import load_dotenv

load_dotenv()

# Configure logging to save logs to a file
log_file_path = os.path.join('logs', 'hf_logs.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'  # Append to the log file
)

class HeartFailure:
    def __init__(self, use_pretrained=True):

        # TODO
        self.hf_model_path = os.getenv('HF_MODEL', './trained-model/hf_version_1.pkl')
        self.hf_test_size = float(os.getenv('HF_TEST_SIZE', '0.25'))
        self.hf_random_state = int(os.getenv('HF_RANDOM_STATE', '44'))
        self.hf_n_estimators = int(os.getenv('HF_N_ESTIMATORS', '10'))
        self.hf_max_depth = int(os.getenv('HF_MAX_DEPTH', '8'))
        self.hf_criterion = os.getenv('HF_CRITERION', 'gini')

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

    def load_data(self):
        df = pd.read_csv('./data-collection/Heart_failure/heart_failure_clinical_records_dataset.csv')
        self.df = df

    def process_data(self):
        X = self.df.drop(['DEATH_EVENT'], axis=1)
        y = self.df['DEATH_EVENT']

        # Splitting the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.hf_test_size, random_state=self.hf_random_state, shuffle=True)

    def model(self):
        rf_model = RandomForestClassifier(n_estimators=self.hf_n_estimators, 
                                          max_depth=self.hf_max_depth, 
                                          criterion=self.hf_criterion, 
                                          random_state=0)
        self.rf_model = rf_model

    def train_model(self):
        self.rf_model.fit(self.X_train, self.y_train)

    def save_model(self):
        filename = './trained-model/hf_version_1.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.rf_model, file)
        logging.info("Model saved successfully.")

    def load_model(self):
        filename = self.hf_model_path
        with open(filename, 'rb') as file:
            self.rf_model = pickle.load(file)
        logging.info("Model loaded successfully.")

    def evaluate(self):
        # Make predictions on the test set
        y_pred_rf = self.rf_model.predict(self.X_test)

        # Calculate the accuracy
        accuracy = accuracy_score(self.y_test, y_pred_rf)
        # Print the results
        print(f'Accuracy: {accuracy}')

    def predict(self, features):
        # Assuming features is a dictionary with the specified keys
        features_df = pd.DataFrame([features])
        
        # Making prediction
        prediction = self.rf_model.predict(features_df)
        
        return prediction[0]  # Returning the prediction value (0 or 1)

