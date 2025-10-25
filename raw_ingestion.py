import requests
import pandas as pd
import hopsworks
import time, datetime, os
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# --- API Parameters ---
lat, lon = 24.8607, 67.0011  # Karachi
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Check if API keys are loaded
if not API_KEY:
    print("Error: OPENWEATHER_API_KEY not found. Please check your .env file.")
    exit()
if not os.getenv("HOPSWORKS_API_KEY"):
    print("Error: HOPSWORKS_API_KEY not found. Please check your .env file.")
    exit()

end = int(time.time())
start = end - 365 * 24 * 60 * 60  # last 730 days

url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"

# --- Fetch Data ---
print("Fetching air quality data from OpenWeather API...")
print(f"Requesting URL: {url.replace(API_KEY, '***YOUR_API_KEY***')}") # Hide key in log
res = requests.get(url)

# -------------------- CRITICAL ERROR HANDLING --------------------
# Check if the request was successful (HTTP 200)
# If not, the response is NOT JSON and will cause the error you see.
if res.status_code != 200:
    print(f"Error: API request failed with status code {res.status_code}")
    print("This is NOT a JSON response. The server returned:")
    print(f"Response text: {res.text}")
    print("\nPlease check your OPENWEATHER_API_KEY in the .env file.")
    exit()  # Stop the script
# -------------------- END ERROR HANDLING --------------------

# Now it's safe to try parsing the JSON
try:
    data = res.json()
except requests.exceptions.JSONDecodeError:
    print("Error: The server returned a 200 OK status, but the response was NOT valid JSON.")
    print(f"Response text: {res.text}")
    exit()

if "list" not in data:
    print(f"Error: 'list' key not in API response. Response was: {data}")
    exit()

records = []
for item in data["list"]:
    records.append({
        "timestamp": datetime.datetime.utcfromtimestamp(item["dt"]),
        "aqi": item.get("main", {}).get("aqi"),
        "co": item.get("components", {}).get("co"),
        "no": item.get("components", {}).get("no"),
        "no2": item.get("components", {}).get("no2"),
        "o3": item.get("components", {}).get("o3"),
        "so2": item.get("components", {}).get("so2"),
        "pm2_5": item.get("components", {}).get("pm2_5"),
        "pm10": item.get("components", {}).get("pm10"),
        "nh3": item.get("components", {}).get("nh3")
    })

df = pd.DataFrame(records)
df = df.sort_values("timestamp").reset_index(drop=True)

if df.empty:
    print("No records fetched. Exiting.")
    exit()

print(f"Fetched {len(df)} raw records.")

# --- Fix 'aqi' column type for Hopsworks ---
print("Fixing 'aqi' column type to match feature group schema...")
if df['aqi'].isnull().any():
    print(f"Warning: {df['aqi'].isnull().sum()} NaNs detected in 'aqi' column. Filling with 0.")
    df['aqi'] = df['aqi'].ffill()

df['aqi'] = df['aqi'].astype("int64")
print("'aqi' column successfully cast to int.")


# --- Store in Feature Store ---
feature_group = fs.get_or_create_feature_group(
    name="karachi_air_quality_raw",
    version=1,
    primary_key=["timestamp"],
    event_time="timestamp",
    description="Raw air pollution data for Karachi from OpenWeather"
)

print("Inserting raw data into Feature Group: karachi_air_quality_raw...")
feature_group.insert(df, write_options={"wait_for_job": True})
print("Raw data insertion complete. 🚀")