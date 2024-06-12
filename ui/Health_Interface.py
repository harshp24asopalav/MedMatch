import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn

# Load the scalers and models
scaler_h = pickle.load(open('scaler_h.pkl', 'rb'))
model_death_event = pickle.load(open('random_forest_model.pkl', 'rb'))

scaler_a = pickle.load(open('scaler_a.pkl', 'rb'))
model_anaemia = pickle.load(open('rf_model_anaemia.pkl', 'rb'))

# Define feature names
feature_names_death_event = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
                             'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 
                             'sex', 'smoking', 'time']

feature_names_anaemia = ['age', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure',
                         'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT']

# Function to predict death event
def predict_death_event(features):
    features_df = pd.DataFrame([features], columns=feature_names_death_event)
    scaled_features = scaler_h.transform(features_df)
    prediction = model_death_event.predict(scaled_features)
    return prediction[0]

# Function to predict anaemia
def predict_anaemia(features):
    features_df = pd.DataFrame([features], columns=feature_names_anaemia)
    scaled_features = scaler_a.transform(features_df)
    prediction = model_anaemia.predict(scaled_features)
    return prediction[0]

# Main interface
st.title("Heart Failure and Anaemia Prediction")

option = st.selectbox("Select Prediction Model", ("Predict Death Event", "Predict Anaemia"))

if option == "Predict Death Event":
    st.header("Predict Death Event")

    # Input fields for all features except the target 'DEATH_EVENT'
    age = st.number_input("Age", min_value=40, max_value=95)
    anaemia = st.selectbox("Anaemia", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=23, max_value=7861)
    diabetes = st.selectbox("Diabetes", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=14, max_value=80)
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
    platelets = st.number_input("Platelets (platelets/mL)", min_value=25010, max_value=850000)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=9.4)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=114, max_value=148)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    smoking = st.selectbox("Smoking", [0, 1])
    time = st.number_input("Follow-up Period (days)", min_value=4, max_value=285)

    features = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, 
                platelets, serum_creatinine, serum_sodium, sex, smoking, time]

    if st.button("Predict Death Event"):
        prediction = predict_death_event(features)
        if prediction == 1:
            st.warning("The model predicts that the patient might have a death event during the follow-up period.")
        else:
            st.success("The model predicts that the patient is unlikely to have a death event during the follow-up period.")

elif option == "Predict Anaemia":
    st.header("Predict Anaemia")
    # Input fields for all features except the target 'anaemia'
    age = st.number_input("Age", min_value=40, max_value=95)
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=23, max_value=7861)
    diabetes = st.selectbox("Diabetes", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=14, max_value=80)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    platelets = st.number_input("Platelets (platelets/mL)", min_value=25010, max_value=850000)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=9.4)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=114, max_value=148)
    smoking = st.selectbox("Smoking", [0, 1])
    time = st.number_input("Follow-up Period (days)", min_value=4, max_value=285)
    death_event = st.selectbox("Death Event", [0, 1])

    features = [age, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                platelets, serum_creatinine, serum_sodium, sex, smoking, time, death_event]

    if st.button("Predict Anaemia"):
        prediction = predict_anaemia(features)
        if prediction == 1:
            st.warning("The model predicts that the patient might have anaemia.")
        else:
            st.success("The model predicts that the patient is unlikely to have anaemia.")