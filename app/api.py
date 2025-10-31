from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import pandas as pd
from pathlib import Path

app = FastAPI(title="AQI Predictor API")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES_JSON = BASE_DIR / "feature_columns.json"
DATA_CSV = BASE_DIR / "features_data.csv"


class PredictRequest(BaseModel):
    features: dict


def load_model_and_schema():
    if not MODEL_PATH.exists() or not FEATURES_JSON.exists():
        raise FileNotFoundError("Model or feature schema not found. Please run train.py to create 'model.pkl' and 'feature_columns.json'.")
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_JSON, "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols


model, feature_columns = load_model_and_schema()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/latest")
def predict_latest():
    if not DATA_CSV.exists():
        raise HTTPException(status_code=404, detail="features_data.csv not found. Run data pipeline or training script first.")
    df = pd.read_csv(DATA_CSV)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data available in features_data.csv")
    latest = df.iloc[-1]
    # Build input vector in the required column order
    x = []
    for c in feature_columns:
        x.append(float(latest.get(c, 0)))
    pred = model.predict([x])[0]
    return {"prediction": float(pred), "index": int(df.index[-1])}


@app.post("/predict")
def predict(payload: PredictRequest):
    incoming = payload.features
    x = []
    for c in feature_columns:
        try:
            val = incoming.get(c, 0)
            x.append(float(val))
        except Exception:
            x.append(0.0)
    pred = model.predict([x])[0]
    return {"prediction": float(pred)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
