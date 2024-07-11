import pandas as pd 
import warnings
warnings.simplefilter(action="ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import resample
import pickle

class BrainStroke:
    def __init__(self, use_pretrained=True):
        #self.scaler = MinMaxScaler() # if use_pretrained = True then assigned scaler here for predict function
        if use_pretrained:
            self.load_model()
        else:
            self.load_data()
            self.process_data()
            self.model()
            self.train_model()
            self.save_model()
            self.evaluate()

    def load_data(self):
        df = pd.read_csv('./data-collection/Brain_Stroke/healthcare-dataset-stroke-data.csv')
        self.df = df

    def process_data(self):
        self.df = self.df.drop(columns="id")
        self.df = self.df.dropna(subset=['bmi'])
        self.df = self.df.drop(self.df[self.df["gender"]=="Other"].index)
        self.df['gender'] = self.df['gender'].replace({'Male': 0, 'Female': 1})
        self.df['ever_married'] = self.df['ever_married'].replace({'No': 0, 'Yes': 1})
        self.df['Residence_type'] = self.df['Residence_type'].replace({'Urban': 0, 'Rural': 1})

        had_stroke = self.df[self.df["stroke"]==1]
        no_stroke = self.df[self.df["stroke"]==0]
        upsampled_had_stroke = resample(had_stroke, replace=True, n_samples=no_stroke.shape[0], random_state=123)
        upsampled_data = pd.concat([no_stroke, upsampled_had_stroke])

        cols = ['work_type', 'smoking_status']
        dums = pd.get_dummies(upsampled_data[cols], dtype=int)
        self.model_data = pd.concat([upsampled_data, dums], axis=1).drop(columns=cols)

        """self.scaler = MinMaxScaler()
        for col in ['age', 'avg_glucose_level', 'bmi']:
            self.scaler.fit(self.model_data[[col]])
            self.model_data[col] = self.scaler.transform(self.model_data[[col]])
"""
        X = self.model_data.drop(columns="stroke")
        y = self.model_data["stroke"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=7, shuffle=True)

    def model(self):
        self.etc_model = ExtraTreesClassifier()

    def train_model(self):
        self.etc_model.fit(self.X_train, self.y_train)

    def save_model(self):
        filename = './trained-model/bs_version_1.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.etc_model, file)

    def load_model(self):
        filename = './trained-model/bs_version_1.pkl'
        with open(filename, 'rb') as file:
            self.etc_model = pickle.load(file)

    def evaluate(self):
        # Make predictions on the test set
        y_bs_pred = self.etc_model.predict(self.X_test)
        # Calculate the accuracy
        accuracy = accuracy_score(self.y_test, y_bs_pred)
        # Print the results
        print(f'Accuracy: {accuracy}')

    def predict(self, features):
        features_df = pd.DataFrame([features])
        #self.scaler = MinMaxScaler()
        """for col in ['age', 'avg_glucose_level', 'bmi']:
            self.scaler.fit(features_df[[col]])
            features_df[col] = self.scaler.transform(features_df[[col]])
"""
        # if use_pretrained = false
        """# Ensure all required columns are present in the input data
        expected_columns = self.model_data.columns
        features_df = features_df.reindex(columns=expected_columns, fill_value=0)

        # Drop the target column 'stroke' for prediction
        X_predict = features_df.drop(columns=['stroke'])"""

        prediction = self.etc_model.predict(features_df)
        return bool(prediction[0])

"""input_data = {
    'gender': 1,
    'age': 78,
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 0,
    'Residence_type': 0,
    'avg_glucose_level': 130.54,
    'bmi': 20.1,
    'work_type_Govt_job': 0,
    'work_type_Never_worked': 0,
    'work_type_Private': 1,
    'work_type_Self-employed': 0,
    'work_type_children': 0,
    'smoking_status_Unknown': 0,
    'smoking_status_formerly smoked': 0,
    'smoking_status_never smoked': 1,
    'smoking_status_smokes': 0
}

bs = BrainStroke(use_pretrained=False)

prediction = bs.predict(input_data)
print(f'Prediction: {prediction}')"""