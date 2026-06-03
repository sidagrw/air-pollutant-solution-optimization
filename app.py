# app.py

import os
import sys
import json
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=HERE)
CORS(app)

print("[Startup] Loading project modules...")
from config import EnvironmentConfig, SimulationConfig, ControllerConfig
from environment import SurrogateEnv
from data_generator import DataGenerator
from evaluation import Evaluator
from controllers import ThresholdController, PIDController, MPCController, PPOController
print("[Startup] Modules loaded.")

_forecaster = None
_model_path = os.path.join(HERE, "models", "rf_aqi_model.pkl")

def get_forecaster():
    global _forecaster
    if _forecaster is not None:
        return _forecaster
    if os.path.exists(_model_path):
        try:
            from forecaster import AQIForecaster
            _forecaster = AQIForecaster(_model_path)
            print(f"[Startup] RF Forecaster loaded from {_model_path}")
        except Exception as e:
            print(f"[Startup] Could not load forecaster: {e}. Using linear surrogate.")
    else:
        print(f"[Startup] Model not found at {_model_path}. Using linear surrogate.")
    return _forecaster


class LinearSurrogate:
    """fallback forecaster if RF model is missing."""
    def predict(self, current_pm10, ambient_inflow, scrubber_action):
        return max(15.0, current_pm10 + ambient_inflow - 0.03*current_pm10 - 0.15*scrubber_action)


def make_controllers(forecaster, ctrl_cfg):
    fc = forecaster if forecaster else LinearSurrogate()
    return {
        "MPC":       MPCController(forecaster=fc, horizon=7),
        "PID":       PIDController(kp=1.5, ki=0.08, kd=3.0, integral_limit=40.0, alpha=0.2),
        "PPO":       PPOController(model_path=os.path.join(HERE, "fake.zip")),
        "Threshold": ThresholdController(
            hysteresis_high=ctrl_cfg.hysteresis_high,
            hysteresis_low=ctrl_cfg.hysteresis_low
        )
    }


@app.route("/api/sensor")
def api_sensor():
    try:
        sim_cfg = SimulationConfig()
        env_cfg = EnvironmentConfig.load_with_live_telemetry(
            api_key=sim_cfg.openaq_api_key,
            location_id=sim_cfg.bapuji_nagar_id
        )
        default = EnvironmentConfig().initial_pm10
        is_live = env_cfg.initial_pm10 != default
        return jsonify({
            "ok": True,
            "pm10": round(env_cfg.initial_pm10, 2),
            "source": "live" if is_live else "default"
        })
    except Exception as e:
        return jsonify({"ok": False, "pm10": 45.0, "source": "default", "error": str(e)})


@app.route("/api/run_benchmark", methods=["POST"])
def api_run_benchmark():
    try:
        body         = request.get_json(silent=True) or {}
        seeds        = body.get("seeds", [42, 1024, 2026, 777, 1337])
        steps        = int(body.get("steps", 60))
        initial_pm10 = float(body.get("initial_pm10", 45.0))

        forecaster = get_forecaster()
        ctrl_cfg   = ControllerConfig()

        scenarios = {
            "Normal Traffic":   "normal_traffic",
            "Traffic Spike":    "traffic_spike",
            "Pollution Burst":  "pollution_burst",
            "Windy Conditions": "windy_conditions",
            "Low Water":        "low_water_scenario"
        }
        ctrl_names = ["MPC", "PID", "PPO", "Threshold"]

        log = []
        raw = {sc: {c: [] for c in ctrl_names} for sc in scenarios}

        def L(t, m): log.append({"type": t, "msg": m})

        L("head", "=" * 58)
        L("head", "  PM10 BENCHMARK PIPELINE")
        L("head", "=" * 58)
        L("info", f"[Config] seeds={seeds}  steps={steps}  initial_pm10={initial_pm10} ug/m3")
        L("success" if forecaster else "warn",
          "[System] RF Forecaster loaded." if forecaster else "[System] RF model missing — using linear surrogate for MPC.")

        for sc_name, sc_func in scenarios.items():
            L("info", f"--- Scenario: {sc_name} ---")

            for seed in seeds:
                gen     = DataGenerator(seed=seed)
                func    = getattr(gen, sc_func)
                inflows = [float(func(t)) for t in range(steps)]

                ctrls = make_controllers(forecaster, ctrl_cfg)
                for ctrl_name, ctrl in ctrls.items():
                    env_cfg   = EnvironmentConfig(initial_pm10=initial_pm10)
                    env       = SurrogateEnv(env_cfg)
                    evaluator = Evaluator()

                    state = env.reset()
                    ctrl.reset()

                    pm_traj, act_traj = [state], []

                    for t in range(steps):
                        action = ctrl.choose_action(current_pm10=state, ambient_inflow=inflows[t])
                        result = env.step(scrubber_action=action, ambient_pm10_inflow=inflows[t])
                        state  = result["next_state"]
                        evaluator.record(pm10=state, action=action, penalties=result["penalties"])
                        pm_traj.append(float(state))
                        act_traj.append(float(action))

                    metrics = evaluator.summary()
                    raw[sc_name][ctrl_name].append({
                        "seed": seed,
                        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
                        "trajectory": {
                            "pm10":    [round(x, 3) for x in pm_traj],
                            "actions": [round(x, 3) for x in act_traj],
                            "inflows": [round(x, 3) for x in inflows]
                        }
                    })

            for c in ctrl_names:
                avg_v = float(np.mean([r["metrics"]["Total Violations"] for r in raw[sc_name][c]]))
                if avg_v > 0:
                    L("warn", f"    [{c}] avg violations: {avg_v:.2f}")

        # aggregate metrics across seeds
        aggregated = {}
        for sc in scenarios:
            aggregated[sc] = {}
            for c in ctrl_names:
                def agg(key):
                    vals = [r["metrics"][key] for r in raw[sc][c]]
                    return {"mean": round(float(np.mean(vals)), 4),
                            "std":  round(float(np.std(vals)),  4)}
                aggregated[sc][c] = {
                    "Average PM10":     agg("Average PM10"),
                    "RMSE":             agg("RMSE"),
                    "Water Usage":      agg("Water Usage"),
                    "Total Violations": agg("Total Violations")
                }

        L("success", f"[System] Done. {len(seeds)} seeds × {len(scenarios)} scenarios × 4 controllers.")

        return jsonify({"ok": True, "log": log, "aggregated": aggregated, "raw": raw,
                        "meta": {"seeds": seeds, "steps": steps, "initial_pm10": initial_pm10}})

    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        body         = request.get_json(silent=True) or {}
        scenario     = body.get("scenario", "Normal Traffic")
        ctrl_name    = body.get("controller", "MPC")
        seed         = int(body.get("seed", 42))
        steps        = int(body.get("steps", 60))
        initial_pm10 = float(body.get("initial_pm10", 45.0))

        sc_map = {
            "Normal Traffic":   "normal_traffic",
            "Traffic Spike":    "traffic_spike",
            "Pollution Burst":  "pollution_burst",
            "Windy Conditions": "windy_conditions",
            "Low Water":        "low_water_scenario"
        }

        gen     = DataGenerator(seed=seed)
        inflows = [float(getattr(gen, sc_map[scenario])(t)) for t in range(steps)]

        forecaster = get_forecaster()
        ctrl_cfg   = ControllerConfig()
        ctrl       = make_controllers(forecaster, ctrl_cfg)[ctrl_name]

        env_cfg   = EnvironmentConfig(initial_pm10=initial_pm10)
        env       = SurrogateEnv(env_cfg)
        evaluator = Evaluator()

        state = env.reset()
        ctrl.reset()
        pm_traj, act_traj = [state], []

        for t in range(steps):
            action = ctrl.choose_action(current_pm10=state, ambient_inflow=inflows[t])
            result = env.step(scrubber_action=action, ambient_pm10_inflow=inflows[t])
            state  = result["next_state"]
            evaluator.record(pm10=state, action=action, penalties=result["penalties"])
            pm_traj.append(float(state))
            act_traj.append(float(action))

        return jsonify({
            "ok": True,
            "trajectory": {
                "pm10":    [round(x, 3) for x in pm_traj],
                "actions": [round(x, 3) for x in act_traj],
                "inflows": [round(x, 3) for x in inflows]
            },
            "metrics": {k: round(float(v), 4) for k, v in evaluator.summary().items()}
        })

    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


if __name__ == "__main__":
    get_forecaster()   # load at startup, not on first request
    print("\n" + "=" * 50)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    try:
        from waitress import serve
        print("[Server] Using waitress")
        serve(app, host="127.0.0.1", port=5000, threads=4, channel_timeout=300)
    except ImportError:
        print("[Server] waitress not found — run: pip install waitress")
        print("[Server] Falling back to Flask dev server")
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)