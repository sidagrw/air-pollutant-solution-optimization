import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/simulations", tags=["simulations"])

class CSTRInput(BaseModel):
    inj_rate: float  
    sim_mins: int   
    spray_type: int 
    initial_no2: float 
    initial_pm25: float 
    initial_pm10: float 
    wind_speed: float  

def get_cpcb_aqi(pollutant, value):
    """Official Indian CPCB Breakpoints."""
    bp = {
        "pm25": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 1000, 401, 500)],
        "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 1000, 401, 500)],
        "no2":  [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)]
    }
    if pollutant not in bp: return 0
    for (low_c, high_c, low_i, high_i) in bp[pollutant]:
        if low_c <= value <= high_c:
            return ((high_i - low_i) / (high_c - low_c)) * (value - low_c) + low_i
    return value

@router.post("/cstr")
def run_cstr_physics(body: CSTRInput):
    Volume = 200000 
    
    # BOOSTED Reaction Kinetics for more realistic, high-impact scrubbing
    k_no2 = 8.50 if body.spray_type == 1 else 14.50 
    k_pm = 5.50 # Boosted Washout coefficient
    
    c_no2 = body.initial_no2
    c_pm25 = body.initial_pm25
    c_pm10 = body.initial_pm10
    
    timeline = []
    pre_aqi = round(max(get_cpcb_aqi("no2", c_no2), get_cpcb_aqi("pm25", c_pm25), get_cpcb_aqi("pm10", c_pm10)))
    
    for m in range(body.sim_mins + 1):
        # High-Efficiency NO2 Chemical Reaction Drop
        c_no2 = max(2.0, c_no2 - ((body.inj_rate * k_no2 * 10) / (Volume / 10000)))
        
        # High-Efficiency PM Washout
        c_pm25 = max(5.0, c_pm25 - ((body.inj_rate * k_pm * 8) / (Volume / 10000)))
        c_pm10 = max(10.0, c_pm10 - ((body.inj_rate * k_pm * 12) / (Volume / 10000)))
        
        current_aqi = round(max(get_cpcb_aqi("no2", c_no2), get_cpcb_aqi("pm25", c_pm25), get_cpcb_aqi("pm10", c_pm10)))
        
        timeline.append({
            "minute": m,
            "no2": round(c_no2, 2),
            "pm25": round(c_pm25, 2),
            "pm10": round(c_pm10, 2),
            "aqi": current_aqi
        })

    # Calculate exact percentage drop for the UI
    safe_initial_no2 = max(body.initial_no2, 0.1)
    reduction_pct = round(((body.initial_no2 - c_no2) / safe_initial_no2) * 100, 1)

    return {
        "initial_no2": round(body.initial_no2, 2),
        "final_no2": round(c_no2, 2),
        "no2_reduction_pct": reduction_pct,
        "initial_aqi": pre_aqi,
        "final_aqi": current_aqi,
        "timeline": timeline
    }