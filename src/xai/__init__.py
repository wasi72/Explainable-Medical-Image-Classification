"""Explainable AI (XAI) package for chest radiograph classification."""

from src.xai.gradcam import GradCAM, GradCAMPlusPlus
from src.xai.saliency import VanillaSaliency, SmoothGrad
from src.xai.visualizer import (
    create_cam_overlay,
    create_clinical_comparison_panel,
    image_to_base64
)
from src.xai.metrics import pointing_game_hit, cam_energy_in_bbox, cam_iou_with_bbox

__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "VanillaSaliency",
    "SmoothGrad",
    "create_cam_overlay",
    "create_clinical_comparison_panel",
    "image_to_base64",
    "pointing_game_hit",
    "cam_energy_in_bbox",
    "cam_iou_with_bbox"
]
