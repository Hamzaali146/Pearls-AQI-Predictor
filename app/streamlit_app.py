import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES_JSON = BASE_DIR / "feature_columns.json"
DATA_CSV = BASE_DIR / "features_data.csv"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_JSON, "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def main():
    st.title("AQI Predictor — Dashboard")

    if not DATA_CSV.exists() or not MODEL_PATH.exists() or not FEATURES_JSON.exists():
        st.warning("Missing artifacts. Please run `train.py` first to generate model and feature files.")
        return

    model, feature_cols = load_model()

    df = pd.read_csv(DATA_CSV)
    # Try to parse timestamp for plotting if present
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df = df.reset_index(drop=True)
        except Exception:
            pass
    st.subheader("Recent feature rows")
    st.dataframe(df.tail(10))

    # --- Visualizations ---
    st.markdown("---")
    st.header("Visualizations")
    available_features = [c for c in feature_cols if c in df.columns]
    if not available_features:
        st.info("No features available for plotting. Make sure `feature_columns.json` matches `features_data.csv`.")
    else:
        default_selection = [f for f in ["pm2_5", "pm10", "aqi"] if f in available_features][:3]
        to_plot = st.multiselect("Select features to plot (time-series)", options=available_features, default=default_selection)
        if to_plot:
            plot_df = df[to_plot].copy()
            # If timestamp exists, set it as index for nicer x-axis
            if "timestamp" in df.columns:
                plot_df.index = df["timestamp"]
            st.line_chart(plot_df.tail(500))

        # If AQI exists, show predicted vs actual for the test portion (quick view)
        if "aqi" in df.columns:
            st.subheader("AQI: actual (from data) — quick view")
            aqi_plot = df[["aqi"]].copy()
            if "timestamp" in df.columns:
                aqi_plot.index = df["timestamp"]
            st.line_chart(aqi_plot.tail(500))

    # Feature importances (if available)
    st.markdown("---")
    st.header("Model feature importances")
    try:
        import numpy as _np
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            imp_map = {c: float(im) for c, im in zip(feature_cols, importances)}
            # Keep only those present in the dataset
            imp_map = {k: v for k, v in imp_map.items() if k in df.columns}
            if imp_map:
                imp_df = pd.DataFrame.from_dict(imp_map, orient="index", columns=["importance"]).sort_values("importance", ascending=False)
                st.bar_chart(imp_df)
            else:
                st.info("No overlapping features between model and dataset for importances.")
        else:
            st.info("Model does not expose feature_importances_.")
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

    st.subheader("Prediction for latest row")
    latest = df.iloc[-1]
    x = [float(latest.get(c, 0)) for c in feature_cols]
    pred = model.predict([x])[0]
    st.metric(label="Predicted AQI (next hour)", value=round(float(pred), 2))

    st.subheader("Select a row to predict")
    idx = st.number_input("Row index (0-based)", min_value=0, max_value=len(df)-1, value=len(df)-1)
    selected = df.iloc[int(idx)]
    st.write(selected)
    x_sel = [float(selected.get(c, 0)) for c in feature_cols]
    pred_sel = model.predict([x_sel])[0]
    st.metric(label=f"Prediction for row {idx}", value=round(float(pred_sel), 2))

    st.caption("This dashboard loads a saved model (`model.pkl`) and `feature_columns.json` created by `train.py`.")

    # --- New: Predict current AQI using all features and optionally from Hopsworks ---
    st.markdown("---")
    st.header("Predict current AQI")
    use_feature_store = st.checkbox("Fetch latest features from Hopsworks Feature Store (if configured)", value=False)

    latest_row = None
    if use_feature_store:
        try:
            import hopsworks
            project = hopsworks.login()
            fs = project.get_feature_store()
            # attempt version 2 then 1 to be resilient
            fg = None
            for v in (2, 1):
                try:
                    fg = fs.get_feature_group("karachi_air_quality_features", version=v)
                    break
                except Exception:
                    fg = None
            if fg is None:
                st.warning("No feature group 'karachi_air_quality_features' found in Feature Store. Falling back to local CSV.")
            else:
                df_fs = fg.read()
                if df_fs.empty:
                    st.warning("Feature group returned no rows. Falling back to local CSV.")
                else:
                    latest_row = df_fs.iloc[-1]
                    st.success("Fetched latest row from Feature Store.")
        except Exception as e:
            st.warning(f"Could not fetch from Hopsworks: {e}. Falling back to local CSV.")

    if latest_row is None:
        latest_row = df.iloc[-1]

    st.subheader("Current feature values (all)")
    # Display only the features used by the model in order
    feature_display = {c: float(latest_row.get(c, 0)) for c in feature_cols}
    st.json(feature_display)

    if st.button("Predict current AQI"):
        x_curr = [float(latest_row.get(c, 0)) for c in feature_cols]
        pred_curr = model.predict([x_curr])[0]
        st.success(f"Predicted current AQI (next hour): {round(float(pred_curr),2)}")
        # Show timestamp if available
        if "timestamp" in latest_row.index:
            st.caption(f"Source timestamp: {latest_row['timestamp']}")
        elif "timestamp_seconds" in latest_row.index:
            st.caption(f"Source timestamp_seconds: {int(latest_row['timestamp_seconds'])}")


if __name__ == "__main__":
    main()
