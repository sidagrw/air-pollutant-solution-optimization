"""Run sample AQI prediction pipeline."""

from src.data.load_data import load_sample_data
from src.data.preprocess import preprocess_aqi_data
from src.features.build_features import build_features
from src.models.train_random_forest import train_random_forest
from src.evaluation.metrics import regression_metrics


def main():
    df = load_sample_data()
    df = preprocess_aqi_data(df)
    X, y = build_features(df)

    model = train_random_forest(X, y)
    predictions = model.predict(X)

    print("Sample pipeline metrics:")
    print(regression_metrics(y, predictions))


if __name__ == "__main__":
    main()
