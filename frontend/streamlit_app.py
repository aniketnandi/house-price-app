import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="House Price Predictor", page_icon="🏠", layout="centered"
)

st.title("🏠 House Price Predictor")

with st.form("input_form"):
    st.write("Enter features (California Housing schema)")
    MedInc = st.number_input(
        "Median Income (10k USD)", min_value=0.0, value=3.5, step=0.1
    )
    HouseAge = st.number_input("House Age", min_value=0.0, value=20.0, step=1.0)
    AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0, step=0.1)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0, step=0.1)
    Population = st.number_input("Population", min_value=0.0, value=1000.0, step=10.0)
    AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0, step=0.1)
    Latitude = st.number_input("Latitude", value=34.05)
    Longitude = st.number_input("Longitude", value=-118.25)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        pred = resp.json()["prediction"]
        st.success(f"Predicted price (in $100,000s): {pred:.3f}")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
with st.expander("API Status"):
    try:
        h = requests.get(f"{API_URL}/health", timeout=5)
        st.write(h.json())
    except Exception as e:
        st.write(f"Health check failed: {e}")
