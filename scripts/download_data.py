"""Script to download historical weather and air quality data for RVCE."""
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# RVCE Bangalore Coordinates
LAT = 12.9237
LON = 77.4987

# Define paths
TRAIN_DIR = Path("data/processed/train")
TEST_DIR = Path("data/processed/test")
TRAIN_FILE = TRAIN_DIR / "rvce_train.csv"
TEST_FILE = TEST_DIR / "rvce_test.csv"

def fetch_historical_data(days_back=365):
    """Fetch historical weather and AQ data from Open-Meteo."""
    print(f"Fetching {days_back} days of historical data for RVCE...")
    
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)
    
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "Asia/Kolkata"
    }
    
    print(" -> Downloading weather data...")
    w_res = requests.get(weather_url, params=weather_params)
    w_res.raise_for_status()
    w_data = w_res.json()["hourly"]
    
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": "pm10,pm2_5,nitrogen_dioxide",
        "timezone": "Asia/Kolkata"
    }
    
    print(" -> Downloading air quality data...")
    aq_res = requests.get(aq_url, params=aq_params)
    aq_res.raise_for_status()
    aq_data = aq_res.json()["hourly"]
    
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(w_data["time"]),
        "AT (°C)": w_data["temperature_2m"],
        "RH (%)": w_data["relative_humidity_2m"],
        "WS (m/s)": [x / 3.6 if x is not None else 0 for x in w_data["wind_speed_10m"]],
        "RF (mm)": w_data["precipitation"],
        "PM10 (µg/m³)": aq_data["pm10"],
        "PM2.5 (µg/m³)": aq_data["pm2_5"],
        "NO2 (µg/m³)": aq_data["nitrogen_dioxide"]
    })
    
    df = df.dropna()
    df["hour"] = df["Timestamp"].dt.hour
    df["month"] = df["Timestamp"].dt.month
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    
    return df

def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    df = fetch_historical_data(days_back=365)
    
    # Split the data chronologically: 80% train, 20% test
    split_index = int(len(df) * 0.8)
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]
    
    print(f" -> Saving {len(df_train)} records to {TRAIN_FILE}")
    df_train.to_csv(TRAIN_FILE, index=False)
    
    print(f" -> Saving {len(df_test)} records to {TEST_FILE}")
    df_test.to_csv(TEST_FILE, index=False)
    
    print("✅ Data download and split complete!")

if __name__ == "__main__":
    main()