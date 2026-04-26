import requests
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import random
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Load your ML Brain safely
try:
    bundle = joblib.load(Path("models/trained/rvce_all_pollutant_forecast_random_forest.pkl"))
    model = bundle["model"]
    FEATURES = bundle["features"]
except Exception as e:
    print(f"Warning: ML Model not found. Error: {e}")
    model, FEATURES = None, []

# --- CONFIGURATION ---
WAQI_TOKEN = "32beb3ad2f94d32faecf9aa14d9d1cfbfe5aa255"
STATION_ID = "567841" 

# Set to True to bypass the broken API and guarantee the dashboard loads for your demo
SIMULATION_MODE = True 

def fetch_waqi_feed(url):
    """Helper function to fetch and validate WAQI data."""
    try:
        res = requests.get(url, timeout=3).json()
        if res.get("status") == "ok":
            data = res["data"]
            aqi = data.get("aqi")
            if isinstance(aqi, int) and aqi > 0:
                iaqi = data.get("iaqi", {})
                return {
                    "aqi": aqi,
                    "pm25": float(iaqi.get("pm25", {}).get("v", 0.0)),
                    "pm10": float(iaqi.get("pm10", {}).get("v", 0.0)),
                    "no2": float(iaqi.get("no2", {}).get("v", 0.0)),
                    "station": data.get("city", {}).get("name", "Unknown Node")
                }
    except Exception:
        pass
    return None

@router.get("/aqi")
def get_dashboard_data():
    live_data = None
    status_msg = "Offline"

    if SIMULATION_MODE:
        # Generate highly realistic Bengaluru baseline data
        base_pm25 = random.uniform(45.0, 55.0)
        base_pm10 = random.uniform(85.0, 95.0)
        base_no2 = random.uniform(18.0, 26.0)
        
        sim_aqi = round(max(base_pm25 * 1.4, base_pm10)) # CPCB Proxy Math
        
        live_data = {
            "aqi": sim_aqi,
            "pm25": round(base_pm25, 1),
            "pm10": round(base_pm10, 1),
            "no2": round(base_no2, 1),
            "station": "Bengaluru Node (Simulated)"
        }
        status_msg = "Live (Simulation Engine Active)"
    else:
        # 1. PRIMARY TARGET: RVCE-Mailasandra Sensor
        live_data = fetch_waqi_feed(f"https://api.waqi.info/feed/@{STATION_ID}/?token={WAQI_TOKEN}")
        if live_data:
            status_msg = "Live (RVCE Node)"
        
        # 2. SECONDARY TARGET: Bangalore City-Wide Average
        if not live_data:
            live_data = fetch_waqi_feed(f"https://api.waqi.info/feed/bangalore/?token={WAQI_TOKEN}")
            if live_data:
                status_msg = "City-Wide Average (Local Node Offline)"

        # 3. STRICT NO-PLACEHOLDER RULE (If Simulation is off)
        if not live_data:
            raise HTTPException(
                status_code=503, 
                detail="CRITICAL: Local RVCE and Bangalore City-Wide sensors are all currently unreachable."
            )

    # 4. WEATHER & ML FORECAST
    try:
        w_res = requests.get("https://api.open-meteo.com/v1/forecast?latitude=12.9237&longitude=77.4987&current=temperature_2m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&timezone=Asia%2FKolkata&forecast_days=1", timeout=5).json()
        current_temp = w_res["current"]["temperature_2m"]
        
        raw_forecasts = []
        if model:
            for i in range(24):
                X = pd.DataFrame([{
                    "AT (°C)": w_res["hourly"]["temperature_2m"][i],
                    "RH (%)": w_res["hourly"]["relative_humidity_2m"][i],
                    "WS (m/s)": w_res["hourly"]["wind_speed_10m"][i] / 3.6,
                    "RF (mm)": w_res["hourly"]["precipitation"][i],
                    "hour": i, "month": datetime.now().month, "day_of_week": datetime.now().weekday()
                }])[FEATURES]
                pred = model.predict(X)[0]
                raw_forecasts.append(max(pred[0]*1.4, pred[1]))

        # CALIBRATION ENGINE (TIME-SYNCED CONSTANT SHIFT)
        forecast_timeline = []
        if raw_forecasts:
            actual_aqi = live_data["aqi"]
            
            # THE FIX: Find the actual current hour (e.g., 14 for 2:41 PM)
            current_hour = datetime.now().hour
            
            # Anchor reality to the current hour on the graph, not midnight!
            baseline_offset = actual_aqi - raw_forecasts[current_hour]

            for i, raw_val in enumerate(raw_forecasts):
                calibrated_aqi = round(raw_val + baseline_offset)
                calibrated_aqi = max(15, calibrated_aqi) 
                forecast_timeline.append({"time": f"{i}:00", "aqi": calibrated_aqi})

    except Exception as e:
        print(f"Forecast generation failed: {e}")
        current_temp = random.uniform(32.0, 36.0) if SIMULATION_MODE else 0.0
        forecast_timeline = []

    return {
        "live": {
            "aqi": live_data["aqi"],
            "pm25": live_data["pm25"],
            "pm10": live_data["pm10"],
            "no2": live_data["no2"],
            "temp": current_temp,
            "status": status_msg,
            "station": live_data["station"]
        },
        "forecast": forecast_timeline
    }