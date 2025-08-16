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
