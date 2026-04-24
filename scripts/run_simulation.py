"""Run intervention simulation."""

from src.data.load_data import load_sample_data
from src.simulation.intervention_simulator import simulate_green_cover_increase, simulate_traffic_reduction


def main():
    df = load_sample_data()
    green = simulate_green_cover_increase(df, increase_percent=10)
    traffic = simulate_traffic_reduction(df, reduction_fraction=0.20)

    print("Green-cover simulation rows:", len(green))
    print("Traffic-reduction simulation rows:", len(traffic))


if __name__ == "__main__":
    main()
