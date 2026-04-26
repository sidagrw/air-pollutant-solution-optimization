"""Basic pipeline tests."""

from src.data.load_data import load_sample_data
from src.data.preprocess import preprocess_aqi_data
from src.features.build_features import build_features


def test_sample_pipeline_loads():
    df = preprocess_aqi_data(load_sample_data())
    X, y = build_features(df)

    assert len(df) > 0
    assert len(X) == len(y)
