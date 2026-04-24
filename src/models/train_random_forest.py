"""Train a Random Forest AQI model."""

from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X, y):
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model
