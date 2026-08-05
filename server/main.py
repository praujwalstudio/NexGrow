"""NexGrow FastAPI backend.

Exposes the full diagnostic pipeline over HTTP:

    POST /diagnose   multipart image upload -> diagnosis JSON
    GET  /health     server + model readiness
    GET  /models     available model metadata

Run from the project root:

    uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from src.inference.pipeline import DiagnosticPipeline
from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS
from src.utils.weather import WeatherAdvisor

MODEL_PATH = os.getenv("NEXGROW_MODEL", "models/student_model/student.pth")
DETECTOR_PATH = os.getenv("NEXGROW_DETECTOR") or None
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

app = FastAPI(title="NexGrow API", version="0.1.0")

_pipeline = None


class LeafResult(BaseModel):
    bbox: list
    det_conf: float
    disease: str
    disease_idx: int
    severity: str
    severity_idx: int
    confidence: float


class DiagnosisResponse(BaseModel):
    passed: bool
    quality: dict
    leaves: list[LeafResult]
    treatment: dict | None = None


def get_pipeline() -> DiagnosticPipeline | None:
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _pipeline = DiagnosticPipeline(model_path=MODEL_PATH, detector_path=DETECTOR_PATH)
    return _pipeline


def _quality_dict(quality) -> dict:
    return {
        "path": quality.path,
        "is_blurry": quality.is_blurry,
        "is_overexposed": quality.is_overexposed,
        "is_underexposed": quality.is_underexposed,
        "laplacian_var": quality.laplacian_var,
        "mean_brightness": quality.mean_brightness,
        "passed": quality.passed,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": get_pipeline() is not None,
        "detector_loaded": DETECTOR_PATH is not None,
    }


@app.get("/models")
def models():
    return {
        "classifier": MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        "detector": DETECTOR_PATH,
        "diseases": DISEASE_CLASSES,
        "severity_levels": SEVERITY_LEVELS,
    }


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    image: UploadFile = File(...),  # noqa: B008 (FastAPI dependency injection)
    weather: bool = Query(False, description="Include weather-aware treatment advice."),
    lat: float = Query(20.5),
    lon: float = Query(78.9),
):
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available (expected at {MODEL_PATH}).",
        )

    contents = await image.read()
    array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result = pipeline.run(img)

    leaves = [
        LeafResult(
            bbox=list(leaf.bbox),
            det_conf=leaf.det_conf,
            disease=leaf.disease,
            disease_idx=leaf.disease_idx,
            severity=leaf.severity,
            severity_idx=leaf.severity_idx,
            confidence=leaf.confidence,
        )
        for leaf in result.leaves
    ]

    treatment = None
    if weather and leaves:
        advisor = WeatherAdvisor(api_key=WEATHER_API_KEY)
        weather_data = advisor.fetch_weather(lat, lon)
        if weather_data:
            top = max(result.leaves, key=lambda leaf: leaf.confidence)
            treatment = advisor.get_treatment(top.disease, weather_data)

    return DiagnosisResponse(
        passed=result.passed,
        quality=_quality_dict(result.quality),
        leaves=leaves,
        treatment=treatment,
    )
