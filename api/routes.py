"""FastAPI Routes for medical inference, explainability (Grad-CAM), and clinical reporting."""

import io
import time
from datetime import datetime
from typing import Dict, Optional
from PIL import Image
import numpy as np
import torch
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status

from src.models.factory import ModelFactory
from src.data.preprocessor import MedicalImagePreprocessor
from src.xai.gradcam import GradCAM, GradCAMPlusPlus
from src.xai.saliency import VanillaSaliency
from src.xai.visualizer import create_cam_overlay, image_to_base64
from src.utils.report_generator import generate_clinical_html_report
from api.schemas import (
    HealthResponse,
    PredictionResponse,
    PathologyPrediction,
    ExplanationResponse,
    ReportResponse
)

router = APIRouter()

DEFAULT_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

DEFAULT_THRESHOLDS = {c: 0.25 for c in DEFAULT_CLASSES}
# Clinical tuning: calibrated operating thresholds for high sensitivity
DEFAULT_THRESHOLDS["Pneumothorax"] = 0.20
DEFAULT_THRESHOLDS["Pneumonia"] = 0.20
DEFAULT_THRESHOLDS["Consolidation"] = 0.20
DEFAULT_THRESHOLDS["Effusion"] = 0.20
DEFAULT_THRESHOLDS["Edema"] = 0.20

# Global model cache and device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_cache: Dict[str, torch.nn.Module] = {}
_preprocessor = MedicalImagePreprocessor()


def get_or_load_model(model_name: str) -> torch.nn.Module:
    """Retrieves cached model or instantiates new one."""
    model_name = model_name.lower().strip()
    if model_name not in _model_cache:
        model = ModelFactory.create_model(
            model_name=model_name,
            num_classes=14,
            pretrained=True
        )
        model.to(DEVICE)
        model.eval()
        _model_cache[model_name] = model
    return _model_cache[model_name]


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns system status, active PyTorch compute device, and available models."""
    return HealthResponse(
        status="operational",
        version="1.0.0",
        device=str(DEVICE),
        available_models=ModelFactory.SUPPORTED_MODELS
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_xray(
    file: UploadFile = File(...),
    model_name: str = Form("densenet121")
):
    """Generates multi-label disease probabilities for 14 chest radiograph abnormalities."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )

    model = get_or_load_model(model_name)
    tensor, _ = _preprocessor.preprocess_for_inference(pil_img)
    tensor = tensor.to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    predictions = []
    positives = []

    for i, cls_name in enumerate(DEFAULT_CLASSES):
        prob = float(probs[i])
        thresh = DEFAULT_THRESHOLDS.get(cls_name, 0.5)
        is_pos = prob >= thresh

        if prob >= 0.70:
            risk = "High"
        elif prob >= thresh:
            risk = "Moderate"
        else:
            risk = "Low"

        if is_pos:
            positives.append(cls_name)

        predictions.append(
            PathologyPrediction(
                pathology=cls_name,
                probability=round(prob, 4),
                threshold=round(thresh, 2),
                is_positive=is_pos,
                risk_level=risk
            )
        )

    return PredictionResponse(
        model_name=model_name,
        predictions=predictions,
        positive_findings=positives,
        inference_latency_ms=round(latency_ms, 2),
        disclaimer="Research & Educational Decision-Support Prototype. Not for primary diagnosis."
    )


@router.post("/explain", response_model=ExplanationResponse)
async def explain_xray(
    file: UploadFile = File(...),
    model_name: str = Form("densenet121"),
    pathology: Optional[str] = Form(None),
    method: str = Form("gradcam"),
    alpha: float = Form(0.45),
    colormap: str = Form("jet")
):
    """Generates visual explainability heatmaps (Grad-CAM, Grad-CAM++, Saliency)."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    model = get_or_load_model(model_name)
    tensor, display_img = _preprocessor.preprocess_for_inference(pil_img)
    tensor = tensor.to(DEVICE)

    # If no pathology specified, select class with highest predicted probability
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    if pathology and pathology in DEFAULT_CLASSES:
        target_idx = DEFAULT_CLASSES.index(pathology)
        chosen_pathology = pathology
    else:
        target_idx = int(np.argmax(probs))
        chosen_pathology = DEFAULT_CLASSES[target_idx]

    pred_prob = float(probs[target_idx])
    target_layer = ModelFactory.get_gradcam_target_layer(model, model_name)

    method = method.lower()
    if method == "gradcam_plus_plus":
        explainer = GradCAMPlusPlus(model, target_layer)
        heatmap = explainer.generate_heatmap(tensor, target_idx)
    elif method == "saliency":
        explainer = VanillaSaliency(model)
        heatmap = explainer.generate_saliency(tensor, target_idx)
    else:
        explainer = GradCAM(model, target_layer)
        heatmap = explainer.generate_heatmap(tensor, target_idx)

    overlay = create_cam_overlay(display_img, heatmap, alpha=alpha, colormap=colormap)
    overlay_b64 = image_to_base64(overlay)

    return ExplanationResponse(
        model_name=model_name,
        target_pathology=chosen_pathology,
        probability=round(pred_prob, 4),
        method=method,
        heatmap_overlay_base64=overlay_b64,
        disclaimer="Visual heatmaps reflect network gradient activations, not definitive lesion borders."
    )


@router.post("/report", response_model=ReportResponse)
async def generate_report_endpoint(
    file: UploadFile = File(...),
    patient_id: str = Form("PT_DEFAULT_001"),
    age: str = Form("45"),
    gender: str = Form("M"),
    view: str = Form("PA"),
    model_name: str = Form("densenet121")
):
    """Generates structured decision support clinical report with embedded Grad-CAM evidence."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    model = get_or_load_model(model_name)
    tensor, display_img = _preprocessor.preprocess_for_inference(pil_img)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    pred_dict = {DEFAULT_CLASSES[i]: float(probs[i]) for i in range(len(DEFAULT_CLASSES))}
    top_idx = int(np.argmax(probs))

    # Generate Grad-CAM for the highest confidence finding
    target_layer = ModelFactory.get_gradcam_target_layer(model, model_name)
    cam = GradCAM(model, target_layer).generate_heatmap(tensor, top_idx)
    overlay = create_cam_overlay(display_img, cam, alpha=0.45, colormap="jet")

    patient_meta = {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "view_position": view
    }

    html = generate_clinical_html_report(
        patient_info=patient_meta,
        predictions=pred_dict,
        thresholds=DEFAULT_THRESHOLDS,
        original_img=display_img,
        gradcam_img=overlay,
        model_name=model_name
    )

    findings = [c for c, p in pred_dict.items() if p >= DEFAULT_THRESHOLDS.get(c, 0.5)]

    return ReportResponse(
        patient_id=patient_id,
        model_name=model_name,
        findings=findings,
        html_report=html,
        timestamp=datetime.now().isoformat()
    )
