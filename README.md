### Setup to run the project

##### **1. Create a virtual environment, run the following commands from the root folder:**
* Initialize a virtual environment: `python -m venv venv`
* Activate the virtual environment:
- Linux or Mac: source `./venv/bin/activate`
- Windows Powershell: `.\venv\Scripts\Activate.ps1`
* Install the Python packages: `pip install -r requirements.txt`
* Install ipykernel: `python -m ipykernel install --user --name=venv --display-name=venv`

##### **2. To run the code**
* Run the `app.py` file from root folder so, this will going to start my server for all Restful APIs
* Then start another terminal and go to 'frontend' folder by this command: `cd .\frontend\`
* `streamlit run interface.py` after typing this command my interface will be open.

##### **3. Check SOA services contract per ML Use case using POSTMAN**

##### **Check the pipeline services via API call**

**i. Heart Failure**
* Run the `heart_failure_app.py`, which will going to start my server
* using `POSTMAN` call the API like below using `POST` method and send the request and get the response,
- 1. http://localhost:5000/hf/load_data
- 2. http://localhost:5000/hf/process_data
- 3. http://localhost:5000/hf/model
- 4. http://localhost:5000/hf/train_model
- 5. http://localhost:5000/hf/save_model
- 6. http://localhost:5000/hf/evaluate

**ii. Brain Stroke**
* Run the `brain_stroke_app.py`, which will going to start my server
* using `POSTMAN` call the API like below using `POST` method and send the request and get the response,
- 1. http://localhost:5000/bs/load_data
- 2. http://localhost:5000/bs/process_data
- 3. http://localhost:5000/bs/model
- 4. http://localhost:5000/bs/train_model
- 5. http://localhost:5000/bs/save_model
- 6. http://localhost:5000/bs/evaluate

**iii. Diabetes**
* Run the `diabetes_app.py`, which will going to start my server
* using `POSTMAN` call the API like below using `POST` method and send the request and get the response,
- 1. http://localhost:5000/di/load_data
- 2. http://localhost:5000/di/process_data
- 3. http://localhost:5000/di/model
- 4. http://localhost:5000/di/train_model
- 5. http://localhost:5000/di/save_model
- 6. http://localhost:5000/di/evaluate

**iv. Kidney Disease**
* Run the `kidney_disease_app.py`, which will going to start my server
* using `POSTMAN` call the API like below using `POST` method and send the request and get the response,
- 1. http://localhost:5000/kd/load_data
- 2. http://localhost:5000/kd/process_data
- 3. http://localhost:5000/kd/model
- 4. http://localhost:5000/kd/train_model
- 5. http://localhost:5000/kd/save_model
- 6. http://localhost:5000/kd/evaluate

**v. Calories Maintaince**
* Run the `cal_pred_app.py`, which will going to start my server
* using `POSTMAN` call the API like below using `POST` method and send the request and get the response,
- 1. http://localhost:5000/cal/load_data
- 2. http://localhost:5000/cal/preprocess_data
- 3. http://localhost:5000/cal/split_data
- 4. http://localhost:5000/cal/train_model
- 5. http://localhost:5000/cal/save_model
- 6. http://localhost:5000/cal/evaluate_model

