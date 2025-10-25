import hopsworks
import pandas as pd
import numpy as np
import warnings

# Suppress warnings from Hopsworks
warnings.filterwarnings("ignore", category=UserWarning, module='hopsworks')

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features from the raw air quality dataframe.
    
    Args:
        df_raw: A pandas DataFrame with the raw data.
        
    Returns:
        A pandas DataFrame with engineered features.
    """
    
    print("Starting feature engineering...")
    
    # 1. Ensure correct data types and sorting
    df = df_raw.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # --- IMPUTATION STEP ---
    print("Handling missing values using time-series interpolation...")
    df = df.set_index('timestamp')
    
    pollutant_cols = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    for col in pollutant_cols:
        original_nan_count = df[col].isnull().sum()
        if original_nan_count > 0:
            print(f"  Imputing {original_nan_count} missing values in '{col}' using linear interpolation.")
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
    df = df.ffill().bfill() 
    df = df.reset_index()
    # --- END IMPUTATION STEP ---

    # 2. Create cyclical time features
    print("Creating cyclical time features...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df = df.drop(columns=['hour', 'day_of_week', 'month'])
    
    # 3. Create Lag Features
    print("Creating lag features...")
    pollutants = ['pm2_5', 'pm10', 'co', 'o3', 'no2', 'so2']
    lags = [1, 3, 24] 

    for col in pollutants:
        for lag in lags:
            df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
            
    # 4. Create Rolling Window Features
    print("Creating rolling window features...")
    windows = [3, 12, 24] 

    for col in pollutants:
        for window in windows:
            df[f'{col}_roll_avg_{window}h'] = df[col].shift(1).rolling(window=window).mean()
            df[f'{col}_roll_std_{window}h'] = df[col].shift(1).rolling(window=window).std()

    # 5. Create Interaction Features
    print("Creating interaction features...")
    df['no_x'] = df['no'] + df['no2']
    df['pm2_5_to_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    
    # 6. Create Target Variables
    print("Creating target variables...")
    df['pm2_5_target_1h'] = df['pm2_5'].shift(-1) 
    df['pm2_5_target_6h'] = df['pm2_5'].shift(-6) 
    df['aqi_target_1h'] = df['aqi'].shift(-1)     
    
    # 7. Clean up
    print(f"Original shape: {df_raw.shape}") # Use df_raw.shape for original
    df = df.dropna()
    print(f"Shape after dropping NaNs: {df.shape}")
    
    df = df.fillna(0)
    
    # --- CHANGE 1: ADD INT PRIMARY KEY ---
    df['timestamp_seconds'] = (df['timestamp'].astype(np.int64) // 1_000_000_000).astype(np.int64)
    
    # -------------------- FIX 1: Fix AQI type --------------------
    # Interpolation converted 'aqi' back to float. We must cast it back to
    # int64 ('bigint') to match the schema.
    df['aqi'] = df['aqi'].astype(np.int64)
    # -------------------------------------------------------------

    # Ensure all data is float for the feature store, except for our keys/int
    for col in df.columns:
        if col not in ['timestamp', 'timestamp_seconds', 'aqi']:
            df[col] = df[col].astype(float)
            
    print("Feature engineering complete.")
    return df

def update_feature_descriptions(feature_group):
    """Adds descriptions to the engineered features in the Hopsworks UI."""
    feature_descriptions = [
        {"name": "timestamp", "description": "Event time of the reading."},
        {"name": "timestamp_seconds", "description": "PRIMARY KEY: Unix timestamp in seconds (bigint)."},
        {"name": "hour_sin", "description": "Sine transformation of the hour (captures daily cycle)."},
        {"name": "pm2_5_lag_1h", "description": "PM2.5 value from 1 hour ago."},
        {"name": "pm2_5_roll_avg_3h", "description": "Rolling average of PM2.5 over the past 3 hours."},
        {"name": "no_x", "description": "Combined Nitrogen Oxides (NO + NO2)."},
        {"name": "pm2_5_target_1h", "description": "TARGET: PM2.5 value 1 hour in the future."},
    ]
    
    print("Updating feature descriptions...")
    for desc in feature_descriptions:
        try:
            feature_group.update_feature_description(desc["name"], desc["description"])
        except Exception as e:
            print(f"Warning: Could not update description for {desc['name']}. {e}")
    print("Feature descriptions updated.")

def main():
    """
    Main function to run the feature engineering pipeline.
    """
    try:
        # 1. Connect to Hopsworks
        project = hopsworks.login()
        fs = project.get_feature_store()
        
        # 2. Get the raw data Feature Group
        try:
            fg_raw = fs.get_feature_group(
                name="karachi_air_quality_raw",
                version=1
            )
            print("Successfully retrieved raw feature group 'karachi_air_quality_raw'.")
        except Exception as e:
            print(f"Error: Could not find 'karachi_air_quality_raw' version 1.")
            print(f"Details: {e}")
            return

        # 3. Read data into a pandas DataFrame
        df_raw = fg_raw.read()
        
        if df_raw.empty:
            print("Raw data is empty. Exiting.")
            return

        # 4. Perform feature engineering
        df_features = engineer_features(df_raw)
        
        if df_features.empty:
            print("No data remaining after feature engineering (likely too little raw data). Exiting.")
            return

        # 5. Get or create the new "features" Feature Group
        fg_engineered = fs.get_or_create_feature_group(
            name="karachi_air_quality_features",
            # -------------------- FIX 2: BUMP THE VERSION --------------------
            # This creates a new feature group (version 2) with your new,
            # correct schema and avoids the mismatch error.
            version=2,
            # -----------------------------------------------------------------
            primary_key=["timestamp_seconds"],  
            event_time="timestamp", 
            description="Engineered features and targets for Karachi air quality prediction.",
            online_enabled=True, 
        )
        
        print("Inserting engineered data into 'karachi_air_quality_features:2'...")
        # 6. Insert the new features
        fg_engineered.insert(
            df_features,
            write_options={"wait_for_job": True} 
        )
        print("Successfully inserted engineered features.")
        
        # 7. Add descriptions
        update_feature_descriptions(fg_engineered)
        
        print("\nFeature engineering pipeline complete! 🚀")
        print(f"Data available in feature group: 'karachi_air_quality_features', version 2.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()