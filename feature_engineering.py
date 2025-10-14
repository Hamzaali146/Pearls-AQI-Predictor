# feature_engineering.py

import hopsworks
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------- Hopsworks Login --------------------
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# -------------------- Load Raw Data --------------------
print("Fetching raw data from Feature Store...")
raw_fg = fs.get_feature_group("karachi_air_quality_raw", version=1)
df = raw_fg.read()
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} records from raw feature group.")

# -------------------- Feature Engineering --------------------
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# Cyclic encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

pollutants = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]

# Rolling window features
for col in pollutants:
    df[f"{col}_rolling_mean_3"] = df[col].rolling(window=3).mean()
    df[f"{col}_rolling_std_3"] = df[col].rolling(window=3).std()

# Lag features
for col in ["aqi", "pm2_5", "pm10"]:
    df[f"{col}_lag_1"] = df[col].shift(1)
    df[f"{col}_lag_3"] = df[col].shift(3)
    df[f"{col}_lag_6"] = df[col].shift(6)

# Ratio features
df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-5)
df["no2_to_o3_ratio"] = df["no2"] / (df["o3"] + 1e-5)

# Drop missing values (caused by rolling/lag)
df = df.dropna().reset_index(drop=True)

# Scale pollutants
scaler = StandardScaler()
df[pollutants] = scaler.fit_transform(df[pollutants])

print("Feature engineering complete.")

# -------------------- Store Engineered Features --------------------
fg_features = fs.get_or_create_feature_group(
    name="karachi_air_quality_features",
    version=1,
    primary_key=["timestamp"],
    description="Engineered air quality features for Karachi (with rolling, lag, cyclic time, ratios)"
)

fg_features.insert(df)
print("Engineered features stored in Feature Group: karachi_air_quality_features")
