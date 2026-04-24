"""Simulation routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/simulations", tags=["simulations"])


@router.get("/")
def list_simulations():
    return {"message": "Simulation endpoint placeholder"}
