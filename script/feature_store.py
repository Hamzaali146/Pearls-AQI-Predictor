import os
import requests
import pandas as pd
import hopsworks
import time, datetime

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

LAT, LON = 24.8607, 67.0011

end = int(time.time())
start = end - 60 * 60  # one hour ago

url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start}&end={end}&appid={OPENWEATHER_API_KEY}"

res = requests.get(url)
data = res.json()

if "list" in data:
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
    if not df.empty:
        feature_group = fs.get_or_create_feature_group(
            name="karachi_air_quality",
            version=1,
            primary_key=["timestamp"],
            description="Hourly air pollution data for Karachi from OpenWeather API"
        )
        feature_group.insert(df)
        print(" Data inserted successfully")
    else:
        print(" No data found for this hour.")
else:
    print(" API returned no valid data:", data)
