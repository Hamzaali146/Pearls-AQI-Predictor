# raw_ingestion.py

import requests
import pandas as pd
import hopsworks
import time, datetime, os
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

lat, lon = 24.8607, 67.0011  # Karachi
API_KEY = os.getenv("OPENWEATHER_API_KEY")

end = int(time.time())
start = end - 90 * 24 * 60 * 60  # last 90 days


url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"

print("Fetching air quality data from OpenWeather API...")
res = requests.get(url)
data = res.json()

records = []
for item in data["list"]:
    records.append({
        "timestamp": datetime.datetime.utcfromtimestamp(item["dt"]),
        "aqi": item["main"]["aqi"],
        "co": item["components"]["co"],
        "no": item["components"]["no"],
        "no2": item["components"]["no2"],
        "o3": item["components"]["o3"],
        "so2": item["components"]["so2"],
        "pm2_5": item["components"]["pm2_5"],
        "pm10": item["components"]["pm10"],
        "nh3": item["components"]["nh3"]
    })

df = pd.DataFrame(records)
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Fetched {len(df)} raw records.")

# -------------------- Store in Feature Store --------------------
feature_group = fs.get_or_create_feature_group(
    name="karachi_air_quality_raw",
    version=1,
    primary_key=["timestamp"],
    description="Raw air pollution data for Karachi from OpenWeather"
)

feature_group.insert(df)
print("Raw data inserted into Feature Group: karachi_air_quality_raw")
