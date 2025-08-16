from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
from pathlib import Path
import structlog
from prometheus_fastapi_instrumentator import Instrumentator
from src.ml.config import settings

logger = structlog.get_logger()
app = FastAPI(title=settings.app_name)
MODEL = None


class HouseFeatures(BaseModel):
    # California Housing feature names
    MedInc: float = Field(..., description="Median income in block group (10k USD)")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class Prediction(BaseModel):
    prediction: float


@app.on_event("startup")
async def load_model():
    global MODEL
    mp = Path(settings.model_path)
    if not mp.exists():
        raise RuntimeError(f"Model not found at {mp}. Train it first.")
    MODEL = joblib.load(mp)
    logger.info("model_loaded", path=str(mp))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
async def predict(payload: HouseFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        import pandas as pd

        df = pd.DataFrame([payload.model_dump()])
        pred = MODEL.predict(df)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        logger.exception("prediction_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))


# Prometheus metrics
if settings.metrics_enabled:
    Instrumentator().instrument(app).expose(app)
