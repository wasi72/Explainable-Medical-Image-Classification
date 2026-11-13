"""Pydantic schemas for REST API endpoints."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    device: str
    available_models: List[str]


class PathologyPrediction(BaseModel):
    pathology: str
    probability: float
    threshold: float
    is_positive: bool
    risk_level: str  # "Low", "Moderate", "High"


class PredictionResponse(BaseModel):
    model_name: str
    predictions: List[PathologyPrediction]
    positive_findings: List[str]
    inference_latency_ms: float
    disclaimer: str


class ExplanationResponse(BaseModel):
    model_name: str
    target_pathology: str
    probability: float
    method: str
    heatmap_overlay_base64: str
    disclaimer: str


class ReportResponse(BaseModel):
    patient_id: str
    model_name: str
    findings: List[str]
    html_report: str
    timestamp: str
