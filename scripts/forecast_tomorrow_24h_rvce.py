import requests
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

MODEL_PATH = Path("models/trained/rvce_all_pollutant_forecast_random_forest.pkl")

OUTPUT_CSV = Path("outputs/predictions/rvce_tomorrow_24h_forecast.csv")
OUTPUT_GRAPH = Path("outputs/figures/rvce_tomorrow_24h_aqi_forecast.png")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_GRAPH.parent.mkdir(parents=True, exist_ok=True)

RVCE_LAT = 12.9237
RVCE_LON = 77.4987

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
TARGETS = bundle["targets"]


def sub_index(c, bp_lo, bp_hi, i_lo, i_hi):
    return ((i_hi - i_lo)/(bp_hi - bp_lo))*(c - bp_lo) + i_lo


def pm25_aqi(x):
    if x <= 30: return sub_index(x,0,30,0,50)
    elif x <= 60: return sub_index(x,31,60,51,100)
    elif x <= 90: return sub_index(x,61,90,101,200)
    elif x <= 120: return sub_index(x,91,120,201,300)
    elif x <= 250: return sub_index(x,121,250,301,400)
    else: return sub_index(min(x,500),251,500,401,500)


def pm10_aqi(x):
    if x <= 50: return sub_index(x,0,50,0,50)
    elif x <= 100: return sub_index(x,51,100,51,100)
    elif x <= 250: return sub_index(x,101,250,101,200)
    elif x <= 350: return sub_index(x,251,350,201,300)
    elif x <= 430: return sub_index(x,351,430,301,400)
    else: return sub_index(min(x,600),431,600,401,500)


def no2_aqi(x):
    if x <= 40: return sub_index(x,0,40,0,50)
    elif x <= 80: return sub_index(x,41,80,51,100)
    elif x <= 180: return sub_index(x,81,180,101,200)
    elif x <= 280: return sub_index(x,181,280,201,300)
    elif x <= 400: return sub_index(x,281,400,301,400)
    else: return sub_index(min(x,1000),401,1000,401,500)


def category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"


def compute_aqi(row):
    subindices = {}

    if "PM2.5 (µg/m³)" in row:
        subindices["PM2.5"] = pm25_aqi(row["PM2.5 (µg/m³)"])

    if "PM10 (µg/m³)" in row:
        subindices["PM10"] = pm10_aqi(row["PM10 (µg/m³)"])

    if "NO2 (µg/m³)" in row:
        subindices["NO2"] = no2_aqi(row["NO2 (µg/m³)"])

    dominant = max(subindices, key=subindices.get)
    return subindices[dominant], dominant


def fetch_weather():
    tomorrow = datetime.now().date() + timedelta(days=1)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": RVCE_LAT,
        "longitude": RVCE_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "Asia/Kolkata",
        "start_date": str(tomorrow),
        "end_date": str(tomorrow),
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(data["time"]),
        "AT (°C)": data["temperature_2m"],
        "RH (%)": data["relative_humidity_2m"],
        "WS (m/s)": [x/3.6 for x in data["wind_speed_10m"]],
        "RF (mm)": data["precipitation"],
    })

    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek

    return df


def main():
    weather = fetch_weather()

    X = weather[FEATURES]
    predictions = model.predict(X)

    output = weather.copy()

    for i, target in enumerate(TARGETS):
        output[target] = predictions[:, i].clip(min=0)

    aqi_vals = []
    cats = []
    dominant = []

    for _, row in output.iterrows():
        aqi, dom = compute_aqi(row)
        aqi_vals.append(aqi)
        cats.append(category(aqi))
        dominant.append(dom)

    output["predicted_aqi"] = aqi_vals
    output["aqi_category"] = cats
    output["dominant_pollutant"] = dominant

    output.to_csv(OUTPUT_CSV, index=False)

    plt.figure(figsize=(12,6))
    plt.plot(output["time"], output["predicted_aqi"], marker="o")
    plt.xticks(rotation=45)
    plt.title("Tomorrow 24 Hour AQI Forecast Near RVCE")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPH)
    plt.close()

    display_cols = ["time","AT (°C)","RH (%)","WS (m/s)","RF (mm)"]
    display_cols += TARGETS
    display_cols += ["predicted_aqi","aqi_category","dominant_pollutant"]

    print("\nTomorrow 24-hour AQI forecast near RVCE:\n")
    print(output[display_cols].round(2).to_string(index=False))

    print("\nSaved CSV:", OUTPUT_CSV)
    print("Saved graph:", OUTPUT_GRAPH)


if __name__ == "__main__":
    main()
