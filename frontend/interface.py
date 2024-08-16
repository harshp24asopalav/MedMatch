import streamlit as st
import requests # For making API requests
from PIL import Image
import tempfile
import logging

# Setup logging
logging.basicConfig(
        filename='/app/logs/app_logs.log',  # Name of the log file
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

# Log application start-up
logger.info("Starting application")

# Define the endpoints for the different disease prediction APIs
API_URLS = {
    "Select a model": None, # Default selection
    "Heart Failure": "http://127.0.0.1:5000/api/is_heart",
    "Diabetes": "http://127.0.0.1:5000/api/is_diabetes",
    "Brain Stroke": "http://127.0.0.1:5000/api/is_stroke",
    "Kidney Disease": "http://127.0.0.1:5000/api/is_kidney",
    "Lung Cancer": "http://127.0.0.1:5000/api/is_lung",
    "Brain Tumor": "http://127.0.0.1:5000/api/is_tumor"
}

# Create a sidebar for choosing disease prediction models
st.sidebar.title("Disease Prediction")
disease_option = st.sidebar.selectbox("Choose a disease prediction model", list(API_URLS.keys()))
logger.info(f"Selected disease model: {disease_option}")

# Use session state to toggle chat
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False

if 'show_calories' not in st.session_state:
    st.session_state.show_calories = False

# Display chat button only when no disease model is selected
if disease_option == "Select a model":
    if st.button("Chat Bot"):
        st.session_state.show_chat = not st.session_state.show_chat

    if st.session_state.show_chat:
        user_question = st.text_input("Ask any question about diseases:")
        if st.button("Get Answer", key='get_answer'):
            if user_question:
                response = requests.post("http://127.0.0.1:5000/api/get_answer", json={"question": user_question})
                logger.info(f"Chatbot API Response status: {response.status_code}")
                if response.status_code == 200:
                    answers = response.json()['answers']
                    for answer in answers:
                        st.write("* " + answer)
                else:
                    st.error("Error: Could not retrieve an answer. Please try again.")

    if st.button("Calories to Maintain"):
        st.session_state.show_calories = not st.session_state.show_calories

    # Calorie maintenance input form
    if st.session_state.show_calories:
        st.title("Calories to Maintain Weight Prediction")
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, step=0.01)
        gender = st.selectbox("Gender", [0, 1])  # Assuming 0: Female, 1: Male
        BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
        BMR = st.number_input("BMR", min_value=1000, max_value=4000, step=10)
        activity_level = st.number_input("Activity Level", min_value=1.0, max_value=2.0, step=0.1)

        if st.button("Predict Calories"):
            input_data = {
                "age": age,
                "weight(kg)": weight,
                "height(m)": height,
                "gender": gender,
                "BMI": BMI,
                "BMR": BMR,
                "activity_level": activity_level
            }
            response = requests.post("http://127.0.0.1:5000/api/calorie_maintenance", json=input_data)
            logger.info(f"Calorie API response status: {response.status_code}")

            if response.status_code == 200:
                calorie_maintenance_level = response.json()['Calorie Maintenance Level']
                st.success(f"The estimated calories to maintain weight are: {calorie_maintenance_level}")
            else:
                st.error("Error: Could not get a response from the API. Please try again.")




if disease_option != "Select a model": # Check if a disease model is selected

    # Define the input form for Heart Failure prediction
    if disease_option == "Heart Failure":
        st.title("Heart Failure Prediction")

        age = st.number_input("Age", min_value=40, max_value=95, step=1)
        anaemia = st.selectbox("Anaemia", [0, 1])
        creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=23, max_value=7861, step=1)
        diabetes = st.selectbox("Diabetes", [0, 1])
        ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=14, max_value=80, step=1)
        high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
        platelets = st.number_input("Platelets (platelets/mL)", min_value=25010, max_value=850000, step=1)
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=9.4, step=0.1)
        serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=114, max_value=148, step=1)
        sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
        smoking = st.selectbox("Smoking", [0, 1])
        time = st.number_input("Follow-up Period (days)", min_value=4, max_value=285, step=1)

        # When the user clicks the 'Predict' button, send the input data to the API
        if st.button("Predict"):
            input_data = {
                "age": age,
                "anaemia": anaemia,
                "creatinine_phosphokinase": creatinine_phosphokinase,
                "diabetes": diabetes,
                "ejection_fraction": ejection_fraction,
                "high_blood_pressure": high_blood_pressure,
                "platelets": platelets,
                "serum_creatinine": serum_creatinine,
                "serum_sodium": serum_sodium,
                "sex": sex,
                "smoking": smoking,
                "time": time
            }

            # Make a POST request to the selected API
            response = requests.post(API_URLS[disease_option], json=input_data)
            logger.info(f"Heart Failure API response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                if result['is_heart']:
                    st.success("The patient is at risk of heart failure.")
                else:
                    st.success("The patient is not at risk of heart failure.")
            else:
                st.error("Error: Could not get a response from the API. Please try again.")


    # Define the input form for Diabetes prediction
    elif disease_option == "Diabetes":
        st.title("Diabetes Prediction")

        gender = st.selectbox("Gender", ['Male', 'Female'])
        age = st.number_input("Age", min_value=0, max_value=80, step=1)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        smoking_history = st.selectbox("Smoking History", ['never', 'No Info', 'current', 'former', 'ever', 'not current'])
        bmi = st.number_input("BMI", min_value=10.0, max_value=95.7, step=0.1)
        hba1c_level = st.number_input("HbA1c Level", min_value=3.5, max_value=9.0, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=80, max_value=300, step=1)

        # When the user clicks the 'Predict' button, send the input data to the API
        if st.button("Predict"):
            input_data = {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": smoking_history,
                "bmi": bmi,
                "HbA1c_level": hba1c_level,
                "blood_glucose_level": blood_glucose_level
            }

            # Make a POST request to the selected API
            response = requests.post(API_URLS[disease_option], json=input_data)
            logger.info(f"Diabetes API response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                if result['diabetes_prediction']:
                    st.success("The patient is at risk of diabetes.")
                else:
                    st.success("The patient is not at risk of diabetes.")
            else:
                st.error("Error: Could not get a response from the API. Please try again.")

    # Define the input form for Brain Stroke prediction
    elif disease_option == "Brain Stroke":
        st.title("Brain Stroke Prediction")

        gender = st.selectbox("Gender (0: Male, 1: Female)", [0, 1])
        age = st.number_input("Age", min_value=0, max_value=82, step=1)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        ever_married = st.selectbox("Has the patient ever been married?", [0, 1])
        residence_type = st.selectbox("Residence Type (0: Urban, 1: Rural)", [0, 1])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=55.12, max_value=271.74, step=0.1)
        bmi = st.number_input("BMI", min_value=10.30, max_value=97.60, step=0.1)
        work_type_govt_job = st.selectbox("Work Type - Govt Job", [0, 1])
        work_type_never_worked = st.selectbox("Work Type - Never Worked", [0, 1])
        work_type_private = st.selectbox("Work Type - Private", [0, 1])
        work_type_self_employed = st.selectbox("Work Type - Self Employed", [0, 1])
        work_type_children = st.selectbox("Work Type - Children", [0, 1])
        smoking_status_unknown = st.selectbox("Smoking Status - Unknown", [0, 1])
        smoking_status_formerly_smoked = st.selectbox("Smoking Status - Formerly Smoked", [0, 1])
        smoking_status_never_smoked = st.selectbox("Smoking Status - Never Smoked", [0, 1])
        smoking_status_smokes = st.selectbox("Smoking Status - Smokes", [0, 1])
        
        # When the user clicks the 'Predict' button, send the input data to the API
        if st.button("Predict"):
            input_data = {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "Residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "work_type_Govt_job": work_type_govt_job,
                "work_type_Never_worked": work_type_never_worked,
                "work_type_Private": work_type_private,
                "work_type_Self-employed": work_type_self_employed,
                "work_type_children": work_type_children,
                "smoking_status_Unknown": smoking_status_unknown,
                "smoking_status_formerly smoked": smoking_status_formerly_smoked,
                "smoking_status_never smoked": smoking_status_never_smoked,
                "smoking_status_smokes": smoking_status_smokes
            }
            # Make a POST request to the selected API
            response = requests.post(API_URLS[disease_option], json=input_data)
            logger.info(f"Brain Stroke API response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                if result['is_stroke']:
                    st.success("The patient is at risk of brain stroke.")
                else:
                    st.success("The patient is not at risk of brain stroke.")
            else:
                st.error("Error: Could not get a response from the API. Please try again.")

    # Kidney Disease Prediction
    elif disease_option == "Kidney Disease":
        st.title("Kidney Disease Prediction")

        Bp = st.number_input("Blood Pressure (Bp)", min_value=50, max_value=180)
        Sg = st.number_input("Specific Gravity (Sg)", min_value=1.005, max_value=1.025)
        Al = st.number_input("Albumin (Al)", min_value=0.0, max_value=5.0)
        Su = st.number_input("Sugar (Su)", min_value=0.0, max_value=5.0)
        Rbc = st.selectbox("Red Blood Cell present or not (Rbc)", [0, 1])
        Bu = st.number_input("Blood Urea (Bu)", min_value=1.5, max_value=391.0)
        Sc = st.number_input("Serum Creatinine (Sc)", min_value=0.4, max_value=76.0)
        Sod = st.number_input("Sodium (Sod)", min_value=4.5, max_value=163.0)
        Pot = st.number_input("Potassium (Pot)", min_value=2.5, max_value=47.0)
        Hemo = st.number_input("Hemoglobin (Hemo)", min_value=3.1, max_value=17.80)
        Wbcc = st.number_input("White Blood Cell Count (Wbcc)", min_value=2200.0, max_value=26400.0)
        Rbcc = st.number_input("Red Blood Cell Count (Rbcc)", min_value=2.1, max_value=8.0)
        Htn = st.selectbox("Hypertension (Htn)", [0, 1])

        if st.button("Predict Kidney Disease"):
            input_data = {
                "Bp": Bp, "Sg": Sg, "Al": Al, "Su": Su, "Rbc": Rbc,
                "Bu": Bu, "Sc": Sc, "Sod": Sod, "Pot": Pot, "Hemo": Hemo,
                "Wbcc": Wbcc, "Rbcc": Rbcc, "Htn": Htn
            }
            # Make a POST request to the selected API
            response = requests.post(API_URLS[disease_option], json=input_data)
            logger.info(f"Kidney Disease API response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                if result['is_kidney_disease']:
                    st.success("The patient is at risk of Kidney Disease.")
                else:
                    st.success("The patient is not at risk of Kidney Disease.")
            else:
                st.error("Error: Could not get a response from the API. Please try again.")

    # Lung Cancer prediction
    elif disease_option == 'Lung Cancer':
        st.title('Lung Cancer Prediction')
        
        # File uploader for the image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            if st.button('Predict'):
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                # Make a prediction request to the Lung Cancer API
                url = API_URLS["Lung Cancer"]
                files = {'file': open(temp_file_path, 'rb')}
                response = requests.post(url, files=files)
                logger.info(f"Lung Cancer API response status: {response.status_code}")
                
                # Handle the response
                if response.status_code == 200:
                    prediction = response.json().get('predicted_class')
                    if prediction == 0:
                        st.success("The Patient may suffering from Adenocarcinoma")
                    elif prediction == 1:
                        st.success("The Patient may suffering from Large Cell Carcinoma")
                    elif prediction == 2:
                        st.success("Normal Lung")
                    elif prediction == 3:
                        st.success("The Patient may suffering from Squamous Cell Carcinoma")
                    else:
                        st.error("Unexpected prediction value.")
                else:
                    st.error("Error in prediction. Please try again.")

    # Brain Tumor prediction
    elif disease_option == 'Brain Tumor':
        st.title('Brain Tumor Prediction')
        
        # File uploader for the image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            if st.button('Predict'):
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                # Make a prediction request to the Brain Tumor API
                url = API_URLS["Brain Tumor"]
                files = {'image': open(temp_file_path, 'rb')}
                response = requests.post(url, files=files)
                logger.info(f"Brain Tumor API response status: {response.status_code}")
                
                # Handle the response
                if response.status_code == 200:
                    prediction = response.json().get('is_tumor')
                    if prediction==1:
                        st.success("The model predicts: Tumor Detected")
                    else:
                        st.success("The model predicts: No Tumor Detected")
                else:
                    st.error("Error in prediction. Please try again.")

# Ensure to keep this at the bottom of your file
if __name__ == "__main__":
    st.markdown( """<style>
        .reportview-container .main .block-container{
            max-width: 800px;
        }
        </style>""", unsafe_allow_html=True)
    logger.info("Application shutdown")