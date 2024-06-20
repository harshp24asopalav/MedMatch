# MedMatch is designed to improve the accuracy of already existing diagnosis industry by introducting required features and by providing an user-friendly interface.
Team Members:
Harsh,
Vishv,
Riddhi,
Tavleen

**Test Data Upload**  
curl -X POST -F "file=@path_to_csv.csv" -F "description=Training data for disease prediction model" http://localhost:63775/upload-training-data

**Test Data Preprocess**
curl -X POST http://localhost:63775/preprocess-data -H "Content-Type: application/json" -d '{"rawDataId": "csv_name_without_extension"}'

**Test Train model**
curl -X POST http://localhost:63775/train_model -H "Content-Type: application/json" -d '{"cleanDataId": "csv_name_without_extension"}'