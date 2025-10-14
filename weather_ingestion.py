import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

FEATURE_GROUP_NAME = "karachi_weather_raw"
FEATURE_GROUP_VERSION = 1

LAT, LON = 24.8607, 67.0011

end_date = datetime.utcnow()
start_date = end_date - timedelta(days=30)


print("Fetching weather data from Open-Meteo...")

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={start_date.strftime('%Y-%m-%d')}"
    f"&end_date={end_date.strftime('%Y-%m-%d')}"
    "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
    "apparent_temperature,precipitation,rain,pressure_msl,cloud_cover,"
    "wind_speed_10m,wind_direction_10m&timezone=auto"
)

response = requests.get(url)
response.raise_for_status()
data = response.json()


df = pd.DataFrame({
    "timestamp": data["hourly"]["time"],
    "temperature": data["hourly"]["temperature_2m"],
    "humidity": data["hourly"]["relative_humidity_2m"],
    "dew_point": data["hourly"]["dew_point_2m"],
    "apparent_temp": data["hourly"]["apparent_temperature"],
    "precipitation": data["hourly"]["precipitation"],
    "rain": data["hourly"]["rain"],
    "pressure": data["hourly"]["pressure_msl"],
    "cloud_cover": data["hourly"]["cloud_cover"],
    "wind_speed": data["hourly"]["wind_speed_10m"],
    "wind_dir": data["hourly"]["wind_direction_10m"],
})

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Fetched {len(df)} hourly records from {start_date.date()} to {end_date.date()}")


print("Connecting to Hopsworks...")

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()


feature_group = fs.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="Raw hourly weather data from Open-Meteo for Karachi",
    primary_key=["timestamp"],
    online_enabled=False
)


print("Inserting data into Feature Group...")
feature_group.insert(df)
print("✅ Weather data inserted successfully!")

print(f"Explore it here: {project.get_url()}/fs/{fs._id}/fg/{feature_group._id}")
