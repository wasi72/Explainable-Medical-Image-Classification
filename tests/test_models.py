"""Tests for model architectures, forward passes, and layer extraction."""

import pytest
import torch
from src.models.factory import ModelFactory


@pytest.mark.parametrize("model_name", ["baseline_cnn", "resnet50", "densenet121", "efficientnet_b0"])
def test_model_forward_pass(model_name):
    # Test on lightweight untrained instances
    model = ModelFactory.create_model(model_name, num_classes=14, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 14), f"Expected shape (2, 14), got {logits.shape} for {model_name}"


def test_gradcam_target_layers():
    for model_name in ["baseline_cnn", "resnet50", "densenet121", "efficientnet_b0"]:
        model = ModelFactory.create_model(model_name, num_classes=14, pretrained=False)
        target = ModelFactory.get_gradcam_target_layer(model, model_name)
        assert target is not None, f"Failed to get target layer for {model_name}"
