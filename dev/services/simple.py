import requests

url = 'http://localhost:54678/upload-training-data'
files = {'file': open('F:/AAIML/INFO8665 Project/MedMatch/heart_failure_clinical_records_dataset.csv', 'rb')}
data = {'description': 'Training data for disease prediction model'}

response = requests.post(url, files=files, data=data)
print(response.json())
