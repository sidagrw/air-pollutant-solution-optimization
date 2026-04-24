"""FastAPI application."""

from fastapi import FastAPI
from backend.app.api.routes import health, predictions, simulations

app = FastAPI(title="AQI Prediction and Simulation API")

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(simulations.router)
