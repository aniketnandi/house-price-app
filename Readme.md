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

Example 1:

<img width="1906" height="883" alt="image" src="https://github.com/user-attachments/assets/fae30276-a31e-4044-8134-cddc31bfda66" />
<img width="1908" height="881" alt="image" src="https://github.com/user-attachments/assets/9fbf0ac5-4d0c-4822-bcd2-07b2212b9d91" />

Example 2:

<img width="1913" height="882" alt="image" src="https://github.com/user-attachments/assets/aa2d6a4b-240d-4dce-bc4a-eae1c2b8d2c5" />
<img width="1909" height="884" alt="image" src="https://github.com/user-attachments/assets/84cd35b9-c4bc-41e0-bfaf-3bae3cbebf10" />
