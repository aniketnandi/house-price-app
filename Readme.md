# House Price App (FastAPI + Streamlit)
 Python-only full-stack ML app to predict house prices.
 ## Quickstart
 ```bash
 python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
 pip install -r requirements.txt
 python -m src.ml.train --model-path models/model_v1.joblib
 uvicorn api.main:app --reload --port 8000
 streamlit run frontend/streamlit_app.py
```
Backend is at:  http://localhost:8000/docs
Frontend is at:  http://localhost:8501
***Screenshots***

<img width="1910" height="884" alt="image" src="https://github.com/user-attachments/assets/989710e6-d94f-48eb-a61b-0828f7e6f6c9" />
<img width="1913" height="885" alt="image" src="https://github.com/user-attachments/assets/4505ebc6-ceba-4c3c-b440-a08b16d71a2f" />
