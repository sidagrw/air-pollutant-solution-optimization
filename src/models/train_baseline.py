"""Train a baseline AQI prediction model."""

from sklearn.linear_model import LinearRegression


def train_linear_regression(X, y):
    """Train baseline Linear Regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model
