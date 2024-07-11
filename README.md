### Setup to run the project

**1. Create a virtual environment, run the following commands from the root folder:**
* Initialize a virtual environment: `python -m venv myvenv`
* Activate the virtual environment:
- Linux or Mac: source `./myvenv/bin/activate`
- Windows Powershell: `.\myvenv\Scripts\Activate.ps1`
* Install the Python packages: `pip install -r requirements.txt`
* Install ipykernel: `python -m ipykernel install --user --name=myvenv --display-name=myvenv`

**2. To run the code**
* Run the app.py file from 'backend' folder so, this will going to start my server for all Restful APIs
* Then start another terminal and go to 'frontend' folder by this command: `cd .\frontend\`
* `streamlit run interface.py` after typing this command my interface will be open.