# Pearls AQI Predictor — Local demo

This workspace contains training, a prediction API, and two simple frontends (Streamlit and Gradio) for the AQI predictor.

Files added/changed by this update:
- `train.py` — trains RandomForest, saves `model.pkl` and `feature_columns.json`, falls back to local CSV if Feature Store unavailable.
- `app/api.py` — FastAPI app exposing `/predict` (POST) and `/predict/latest` (GET).
- `app/streamlit_app.py` — Streamlit dashboard that shows recent rows and predictions.
- `app/gradio_app.py` — Gradio demo that accepts JSON feature mappings and returns a prediction.
- `requirements.txt` — appended packages required for running the apps.

Quick start (local):

1) Create / activate Python environment (Windows PowerShell):

```powershell
python -m venv myenv; .\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Train and save a model (this will try Hopsworks first; if unavailable it will use `features_data.csv`):

```powershell
python train.py
```

This produces `model.pkl`, `feature_columns.json`, and re-writes `features_data.csv` in the repo root.

3) Run the FastAPI prediction API:

```powershell
# From project root
uvicorn app.api:app --reload --port 8000
```

Endpoints:
- `GET /health` — health check
- `GET /predict/latest` — predict using the last row from `features_data.csv`
- `POST /predict` — accept JSON body {"features": {"pm2_5": 12.3, ...}}

4) Run the Streamlit dashboard:

```powershell
streamlit run app/streamlit_app.py
```

5) Run the Gradio demo:

```powershell
python app/gradio_app.py
```

Notes & next steps:
- The model expects the same set of features recorded in `feature_columns.json`. The training script writes that file.
- If you plan to deploy, consider adding authentication to the API and locking down the Hopsworks credentials.
- Optionally convert the Streamlit app to call the API (instead of loading local `model.pkl`) for a clearer separation between frontend and model service.
