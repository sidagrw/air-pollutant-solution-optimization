"""Prediction routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/")
def list_predictions():
    return {"message": "Prediction endpoint placeholder"}
