import hopsworks
import pandas as pd
import os

PROJECT_NAME = "pearls_aqi_predictor"
AIR_RAW_FG = "karachi_air_quality_raw"
WEATHER_RAW_FG = "karachi_weather_raw"
FEATURE_FG = "karachi_air_quality_features"

print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

print("Fetching raw feature groups...")
air_df = fs.get_feature_group(AIR_RAW_FG, version=1).read()
weather_df = fs.get_feature_group(WEATHER_RAW_FG, version=1).read()

print(f"Air data: {air_df.shape} | Weather data: {weather_df.shape}")

# --- Time alignment ---
air_df["timestamp"] = pd.to_datetime(air_df["timestamp"]).dt.round("H")
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"]).dt.round("H")

merged_df = pd.merge_asof(
    air_df.sort_values("timestamp"),
    weather_df.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("1h")
).dropna()

# --- Feature engineering ---
merged_df["hour"] = merged_df["timestamp"].dt.hour
merged_df["day"] = merged_df["timestamp"].dt.day
merged_df["month"] = merged_df["timestamp"].dt.month
merged_df["dayofweek"] = merged_df["timestamp"].dt.dayofweek
merged_df["is_weekend"] = (merged_df["dayofweek"] >= 5).astype(int)

import numpy as np
merged_df["hour_sin"] = np.sin(2 * np.pi * merged_df["hour"] / 24)
merged_df["hour_cos"] = np.cos(2 * np.pi * merged_df["hour"] / 24)
merged_df["day_sin"] = np.sin(2 * np.pi * merged_df["day"] / 31)
merged_df["day_cos"] = np.cos(2 * np.pi * merged_df["day"] / 31)

for col in ["co","no","no2","o3","so2","pm2_5","pm10","nh3"]:
    merged_df[f"{col}_rolling_mean_3"] = merged_df[col].rolling(3, min_periods=1).mean()
    merged_df[f"{col}_rolling_std_3"] = merged_df[col].rolling(3, min_periods=1).std()

for col in ["aqi","pm2_5","pm10"]:
    for lag in [1,3,6]:
        merged_df[f"{col}_lag_{lag}"] = merged_df[col].shift(lag)

merged_df["no2_to_o3_ratio"] = merged_df["no2"] / (merged_df["o3"] + 1e-6)
merged_df["no2_to_so2"] = merged_df["no2"] / (merged_df["so2"] + 1e-6)
merged_df["temp_humidity_index"] = merged_df["temperature"] * merged_df["humidity"] / 100
merged_df["pressure_change"] = merged_df["pressure"].diff()
merged_df["rolling_pm2_5"] = merged_df["pm2_5"].rolling(3, min_periods=1).mean()
merged_df["rolling_temp"] = merged_df["temperature"].rolling(3, min_periods=1).mean()

merged_df.dropna(inplace=True)
print(f"Final engineered dataset shape: {merged_df.shape}")

try:
    fs.get_feature_group(FEATURE_FG, version=1).delete()
    print(" Old feature group deleted.")
except Exception:
    print("No previous feature group found (fresh creation).")

feature_fg = fs.create_feature_group(
    name=FEATURE_FG,
    version=1,
    description="Combined and engineered weather + air pollution features for AQI prediction",
    primary_key=["timestamp"],
    online_enabled=False
)

feature_fg.insert(merged_df)
print("Engineered features stored successfully!")

print(f"Explore it here: {project.get_url()}/fs/{fs._id}/fg/{feature_fg._id}")
