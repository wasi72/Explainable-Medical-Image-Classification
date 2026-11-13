"""Tests for Explainable AI (Grad-CAM, Grad-CAM++, Saliency, and Localization metrics)."""

import pytest
import numpy as np
import torch

from src.models.baseline_cnn import BaselineCNN
from src.xai.gradcam import GradCAM, GradCAMPlusPlus
from src.xai.saliency import VanillaSaliency
from src.xai.metrics import pointing_game_hit, cam_energy_in_bbox, cam_iou_with_bbox


def test_gradcam_generation():
    model = BaselineCNN(num_classes=14)
    model.eval()
    target_layer = model.get_target_layer()

    explainer = GradCAM(model, target_layer)
    dummy_input = torch.randn(1, 3, 224, 224)

    cam = explainer.generate_heatmap(dummy_input, target_class_idx=1)

    assert cam.shape == (224, 224)
    assert 0.0 <= np.min(cam)
    assert np.max(cam) <= 1.0


def test_gradcam_plus_plus_generation():
    model = BaselineCNN(num_classes=14)
    model.eval()
    target_layer = model.get_target_layer()

    explainer = GradCAMPlusPlus(model, target_layer)
    dummy_input = torch.randn(1, 3, 224, 224)

    cam = explainer.generate_heatmap(dummy_input, target_class_idx=2)

    assert cam.shape == (224, 224)
    assert 0.0 <= np.min(cam)
    assert np.max(cam) <= 1.0


def test_saliency_generation():
    model = BaselineCNN(num_classes=14)
    model.eval()

    explainer = VanillaSaliency(model)
    dummy_input = torch.randn(1, 3, 224, 224)

    saliency = explainer.generate_saliency(dummy_input, target_class_idx=0)

    assert saliency.shape == (224, 224)
    assert 0.0 <= np.min(saliency)
    assert np.max(saliency) <= 1.0


def test_localization_metrics():
    # Heatmap with peak at (100, 100)
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[90:110, 90:110] = 1.0

    bbox = (80, 80, 40, 40)  # (x, y, w, h)
    assert pointing_game_hit(cam, bbox) is True

    outside_bbox = (10, 10, 20, 20)
    assert pointing_game_hit(cam, outside_bbox) is False

    energy = cam_energy_in_bbox(cam, bbox)
    assert energy == 1.0  # All energy is inside bbox
