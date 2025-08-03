uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

deployemnt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1   
    

ngrok http 8000    

python run.py

add
GEMINI_API_KEY=API

to 
.env
in base directory
