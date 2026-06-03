"""
IEEE Publication-Grade Benchmark Pipeline
Benchmarking Classical and AI-Based Controllers for Adaptive Urban Air Pollution Mitigation
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Local Environment and Framework Imports
from config import EnvironmentConfig
from environment import SurrogateEnv
from data_generator import DataGenerator
from evaluation import Evaluator
from forecaster import AQIForecaster
from controllers import (
    ThresholdController,
    PIDController,
    MPCController,
    PPOController
)

# =====================================================================
# PIPELINE ARCHITECTURE
# =====================================================================

class BenchmarkPipeline:
    def __init__(self, seeds: List[int], steps: int = 60, target_pm10: float = 50.0):
        self.seeds = seeds
        self.steps = steps
        self.target_pm10 = target_pm10
        
        # Output directory structure
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"results/benchmark_{self.timestamp}"
        self.dirs = {
            "raw": f"{self.base_dir}/csv",
            "summaries": f"{self.base_dir}/summaries",
            "trajectories": f"{self.base_dir}/trajectories",
            "plots": f"{self.base_dir}/plots"
        }
        self._create_directories()

        # STRICT FORECASTER LOADING
        model_path = "models/rf_aqi_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CRITICAL: Forecaster model not found at {model_path}.")
            
        self.forecaster = AQIForecaster(model_path)
        print(f"[System] Successfully loaded Random Forest Forecaster from {model_path}")

        # Configured based on mathematically calibrated IEEE ranges
        self.controllers = {
            "Threshold": lambda: ThresholdController(),
            "PID": lambda: PIDController(kp=1.5, ki=0.08, kd=3.0, integral_limit=40.0, alpha=0.2),
            "MPC": lambda: MPCController(forecaster=self.forecaster, horizon=12),
            "PPO": lambda: PPOController(model_path="fake.zip") # Swap with real .zip when trained
        }

        # Scenario mapping from DataGenerator
        self.scenarios = {
            "Normal Traffic": "normal_traffic",
            "Traffic Spike": "traffic_spike",
            "Pollution Burst": "pollution_burst",
            "Windy Conditions": "windy_conditions",
            "Low Water": "low_water_scenario"
        }

        # Data stores
        self.raw_metrics = []
        # Structure: trajectories[scenario][controller][seed] = {time_series_data}
        self.trajectories = {s: {c: [] for c in self.controllers} for s in self.scenarios}

    def _create_directories(self):
        """Generates the required file storage structure."""
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)
            
    def save_metadata(self):
        """Logs reproducibility data for the experiment."""
        metadata = {
            "timestamp": self.timestamp,
            "seeds": self.seeds,
            "simulation_steps": self.steps,
            "target_pm10": self.target_pm10,
            "controllers_tested": list(self.controllers.keys()),
            "scenarios_tested": list(self.scenarios.keys())
        }
        with open(f"{self.base_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def run_single_episode(self, controller: Any, inflows: List[float]) -> Tuple[Dict, Dict]:
        """Runs a completely isolated single simulation trajectory."""
        env = SurrogateEnv(EnvironmentConfig())
        evaluator = Evaluator()
        
        state = env.reset()
        controller.reset()
        
        traj_pm10 = [state]
        traj_actions = []
        
        for t in range(self.steps):
            inflow = inflows[t]
            action = controller.choose_action(current_pm10=state, ambient_inflow=inflow)
            
            result = env.step(scrubber_action=action, ambient_pm10_inflow=inflow)
            state = result["next_state"]
            
            evaluator.record(pm10=state, action=action, penalties=result["penalties"])
            
            traj_pm10.append(state)
            traj_actions.append(action)
            
        metrics = evaluator.summary()
        trajectory_data = {
            "pm10": np.array(traj_pm10[:-1]),
            "actions": np.array(traj_actions),
            "inflows": np.array(inflows)
        }
        
        return metrics, trajectory_data

    def execute_benchmark(self):
        """Executes the complete multi-scenario, multi-seed benchmark."""
        print(f"\n{'='*80}")
        print("🚀 EXECUTING PUBLICATION-GRADE CONTROLLER BENCHMARK")
        print(f"{'='*80}\n")
        
        for scenario_name, scenario_func in self.scenarios.items():
            print(f"--- Simulating Scenario: {scenario_name} ---")
            
            for seed in self.seeds:
                # GENERATE IDENTICAL DISTURBANCE FOR FAIRNESS
                gen = DataGenerator(seed=seed)
                func = getattr(gen, scenario_func)
                inflows = [func(t) for t in range(self.steps)]
                
                # RUN CONTROLLERS AGAINST IDENTICAL DISTURBANCE
                for ctrl_name, ctrl_factory in self.controllers.items():
                    ctrl = ctrl_factory() 
                    
                    metrics, traj = self.run_single_episode(controller=ctrl, inflows=inflows)
                    
                    metrics["Scenario"] = scenario_name
                    metrics["Controller"] = ctrl_name
                    metrics["Seed"] = seed
                    self.raw_metrics.append(metrics)
                    self.trajectories[scenario_name][ctrl_name].append(traj)

    def process_and_save_results(self):
        """Aggregates statistical results, exports raw trajectories, and formats LaTeX tables."""
        df_raw = pd.DataFrame(self.raw_metrics)
        df_raw.to_csv(f"{self.dirs['raw']}/all_experiments_raw.csv", index=False)
        
        # 1. Export Raw Trajectory Time-Series Data to the 'trajectories/' folder
        for scenario in self.scenarios:
            for ctrl in self.controllers:
                pm10_matrix = np.array([t['pm10'] for t in self.trajectories[scenario][ctrl]])
                act_matrix = np.array([t['actions'] for t in self.trajectories[scenario][ctrl]])
                inflow_matrix = np.array([t['inflows'] for t in self.trajectories[scenario][ctrl]])

                # Create a clean DataFrame of the averaged time-series
                df_traj = pd.DataFrame({
                    "Time_Step": np.arange(self.steps),
                    "Inflow_Disturbance": inflow_matrix.mean(axis=0),
                    "Mean_PM10": pm10_matrix.mean(axis=0),
                    "Std_PM10": pm10_matrix.std(axis=0),
                    "Mean_Action": act_matrix.mean(axis=0),
                    "Std_Action": act_matrix.std(axis=0)
                })
                
                safe_scen = scenario.replace(' ', '_')
                df_traj.to_csv(f"{self.dirs['trajectories']}/{safe_scen}_{ctrl}_data.csv", index=False)

        # 2. Process High-Level Summary Metrics
        summary_cols = ["Average PM10", "RMSE", "Water Usage", "Total Violations"]
        df_summary = df_raw.groupby(["Scenario", "Controller"])[summary_cols].agg(['mean', 'std']).reset_index()
        
        # Flatten MultiIndex columns
        df_summary.columns = ['_'.join(col).strip('_') for col in df_summary.columns.values]
        
        # Scale Water Usage so the Bar Chart Y-Axis is readable
        df_summary['Water_Usage_kUnits'] = df_summary['Water Usage_mean'] / 1000.0
        
        df_summary.to_csv(f"{self.dirs['summaries']}/statistical_summary.csv", index=False)
        self.df_summary = df_summary 
        
        self._print_console_summary(df_summary)

    def _print_console_summary(self, df: pd.DataFrame):
        """Prints a clean, academic tabular summary to the console."""
        print(f"\n{'='*85}")
        print(f"{'AGGREGATED BENCHMARK RESULTS (N='}{len(self.seeds)}{' seeds)' :<60}")
        print(f"{'='*85}")
        print(f"{'Scenario':<20} | {'Controller':<10} | {'Avg PM10':<8} | {'RMSE':<8} | {'Water(k)':<8} | {'Violations'}")
        print(f"{'-'*85}")
        
        for _, row in df.iterrows():
            print(f"{row['Scenario']:<20} | {row['Controller']:<10} | "
                  f"{row['Average PM10_mean']:<8.2f} | {row['RMSE_mean']:<8.2f} | "
                  f"{row['Water_Usage_kUnits']:<8.1f} | {row['Total Violations_mean']:.2f}")
        print(f"{'='*85}\n")

    def generate_plots(self):
        """Generates IEEE-compliant plots separated into their correct folders."""
        print("Generating publication-quality plots...")
        
        line_styles = {'Threshold': ':', 'PID': '--', 'MPC': '-', 'PPO': '-.'}
        colors = {'Threshold': '#d62728', 'PID': '#1f77b4', 'MPC': '#2ca02c', 'PPO': '#ff7f0e'}
        
        # 1. TIME-SERIES TRAJECTORY PLOTS (Saved to trajectories/ folder)
        for scenario in self.scenarios:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            fig.suptitle(f"Transient Response under {scenario}", fontsize=14, fontweight='bold')
            
            for ctrl in self.controllers:
                pm10_matrix = np.array([t['pm10'] for t in self.trajectories[scenario][ctrl]])
                act_matrix = np.array([t['actions'] for t in self.trajectories[scenario][ctrl]])
                
                pm10_mean = pm10_matrix.mean(axis=0)
                pm10_std = pm10_matrix.std(axis=0)
                act_mean = act_matrix.mean(axis=0)
                
                time = np.arange(len(pm10_mean))
                
                ax1.plot(time, pm10_mean, label=ctrl, linestyle=line_styles[ctrl], color=colors[ctrl], linewidth=2)
                ax1.fill_between(time, pm10_mean - pm10_std, pm10_mean + pm10_std, color=colors[ctrl], alpha=0.1)
                ax2.plot(time, act_mean, label=ctrl, linestyle=line_styles[ctrl], color=colors[ctrl], linewidth=1.5)

            # Using 'r' string to prevent SyntaxWarning for LaTeX
            ax1.axhline(self.target_pm10, color='black', linestyle='-', linewidth=1, label="Target (50)")
            ax1.axhline(100.0, color='red', linestyle='--', linewidth=1, label="Limit (100)")
            ax1.set_ylabel(r"PM10 Concentration ($\mu g/m^3$)", fontsize=11)
            ax1.legend(loc="upper right", ncol=2)
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            ax2.set_xlabel("Time (minutes)", fontsize=11)
            ax2.set_ylabel("Scrubber Effort (%)", fontsize=11)
            ax2.set_ylim(-5, 105)
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            # SAVED TO TRAJECTORIES FOLDER
            safe_scen = scenario.replace(' ', '_')
            plt.savefig(f"{self.dirs['plots']}/{safe_scen}_trajectory.png", dpi=300)
            plt.close()

        # 2. COMPARATIVE BAR CHARTS (Saved to plots/ folder)
        self._generate_bar_chart("RMSE_mean", "RMSE (Tracking Error)", "RMSE_Comparison.png")
        self._generate_bar_chart("Total Violations_mean", "Cumulative Safety Violations", "Violations_Comparison.png")
        self._generate_bar_chart("Water_Usage_kUnits", "Water Usage (kUnits)", "Water_Usage_Comparison.png")

        print(f"Benchmarking complete. All assets saved to: {self.base_dir}/")

    def _generate_bar_chart(self, metric_col: str, ylabel: str, filename: str):
        """Helper to generate grouped bar charts."""
        df_pivot = self.df_summary.pivot(index='Scenario', columns='Controller', values=metric_col)
        
        ax = df_pivot.plot(kind='bar', figsize=(10, 6), colormap='tab10', edgecolor='black')
        ax.set_title(f"Controller Comparison: {ylabel}", fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Disturbance Scenario", fontsize=12)
        plt.xticks(rotation=15)
        plt.legend(title="Controller Type")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.dirs['plots']}/{filename}", dpi=300)
        plt.close()


if __name__ == "__main__":
    experiment_seeds = [42, 1024, 2026, 777, 1337] 
    
    pipeline = BenchmarkPipeline(seeds=experiment_seeds)
    pipeline.save_metadata()
    pipeline.execute_benchmark()
    pipeline.process_and_save_results()
    pipeline.generate_plots()