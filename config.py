# config.py
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    dt: float = 1.0
    decay_rate: float = 0.03
    scrubber_efficiency: float = 0.32
    pm10_limit: float = 100.0
    pm10_floor: float = 15.0
    initial_pm10: float = 45.0

    @classmethod
    def load_with_live_telemetry(cls, api_key: str, location_id: int = 6974):
        try:
            from openaq import OpenAQ          # imported here only, not at module level
            client = OpenAQ(api_key=api_key)
            sensor_metadata = client.locations.sensors(locations_id=location_id)
            sensor_map = {s.id: s.parameter.name.lower() for s in sensor_metadata.results}
            latest_data = client.locations.latest(locations_id=location_id)
            for result in latest_data.results:
                if sensor_map.get(result.sensors_id, "") == "pm10":
                    live = float(result.value)
                    print(f"[Config] Live PM10: {live} ug/m3")
                    return cls(initial_pm10=live)
            print("[Config] PM10 not found in response. Using default.")
            return cls()
        except Exception as e:
            print(f"[Config] API error: {e}. Using default.")
            return cls()


@dataclass(frozen=True)
class SimulationConfig:
    total_timesteps: int = 1440
    random_seed: int = 42
    model_path: str = "models/rf_aqi_model.pkl"
    openaq_api_key: str = "9ee18918b15274daeae8843253082fe15c8726b69bf81e567aaa96b2b52981bc"
    bapuji_nagar_id: int = 6974


@dataclass(frozen=True)
class ControllerConfig:
    mpc_horizon: int = 15
    control_min_bound: float = 0.0
    control_max_bound: float = 100.0
    hysteresis_high: float = 85.0
    hysteresis_low: float = 40.0
