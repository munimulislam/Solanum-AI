Potato Prediction Web App - Solanum AI

dependency install:
pip install --no-cache-dir -r requirements.txt
** Fix pointer to teagasc-potato-trials-data-pipeline in requirements.txt to run this command properly **

run fastapi server:
uvicorn app:app --app-dir src --reload
