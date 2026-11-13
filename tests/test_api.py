"""Tests for FastAPI endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "densenet121" in data["available_models"]


def test_predict_endpoint():
    # Create small in-memory PNG
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/v1/predict",
        files={"file": ("test_xray.png", buf, "image/png")},
        data={"model_name": "baseline_cnn"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "baseline_cnn"
    assert len(data["predictions"]) == 14
    assert "inference_latency_ms" in data


def test_explain_endpoint():
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/v1/explain",
        files={"file": ("test_xray.png", buf, "image/png")},
        data={
            "model_name": "baseline_cnn",
            "method": "gradcam",
            "pathology": "Cardiomegaly"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_pathology"] == "Cardiomegaly"
    assert data["heatmap_overlay_base64"].startswith("data:image/png;base64,")
