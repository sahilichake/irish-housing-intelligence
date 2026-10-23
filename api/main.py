"""REST API for the price model.

Run:   uvicorn api.main:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import COUNTIES
from src.models.predict import load_bundle, make_row, predict

app = FastAPI(title="Irish House Price API", version="1.0")


class PropertyIn(BaseModel):
    county: str = Field(..., examples=["Dublin"])
    is_new: bool = False
    is_apartment: bool = False
    size_band: Literal["unknown", "<38m2", "38-125m2", ">125m2"] = "unknown"
    eircode: str = Field("", examples=["D04 X2Y3"])
    dublin_district: str | None = Field(None, examples=["D4"])
    month: int = Field(6, ge=1, le=12)
    town: str = Field("UNKNOWN", examples=["GREYSTONES"])
    address: str = Field("", examples=["10 Lismore Rd, Kimmage, Dublin 12"])


@app.get("/health")
def health():
    try:
        b = load_bundle()
        return {"status": "ok", "model_trained_until": b["trained_until"]}
    except FileNotFoundError as exc:
        return {"status": "no_model", "detail": str(exc)}


@app.post("/predict")
def predict_price(p: PropertyIn):
    if p.county not in COUNTIES:
        raise HTTPException(422, f"county must be one of {COUNTIES}")
    row = make_row(p.county, p.is_new, p.is_apartment, p.size_band, p.eircode, p.dublin_district, p.month, p.town, p.address)
    return predict(row)
