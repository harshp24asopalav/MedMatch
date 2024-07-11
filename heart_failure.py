import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class HeartFailure:
    def __init__(self, use_pretrained=True):
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=44, shuffle=True)

    def model(self):
        rf_model = RandomForestClassifier(n_estimators=10, max_depth=8, criterion='gini', random_state=0)
        self.rf_model = rf_model

    def train_model(self):
        self.rf_model.fit(self.X_train, self.y_train)

    def save_model(self):
        filename = './trained-model/hf_version_1.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.rf_model, file)

    def load_model(self):
        filename = './trained-model/hf_version_1.pkl'
        with open(filename, 'rb') as file:
            self.rf_model = pickle.load(file)

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
