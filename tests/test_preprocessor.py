"""Tests for medical image preprocessing and quality validation."""

import pytest
import numpy as np
from PIL import Image
import torch

from src.data.preprocessor import MedicalImagePreprocessor


def test_preprocessor_grayscale_to_rgb():
    prep = MedicalImagePreprocessor(image_size=(224, 224))
    # Synthetic grayscale image
    gray_img = Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8))
    rgb_img = prep.harmonize_channels(gray_img)

    assert rgb_img.mode == "RGB"
    assert rgb_img.size == (300, 300)


def test_preprocessor_clahe():
    prep = MedicalImagePreprocessor(use_clahe=True)
    arr = np.random.randint(40, 180, (224, 224), dtype=np.uint8)
    enhanced = prep.apply_clahe(arr)

    assert enhanced.shape == (224, 224)
    assert enhanced.dtype == np.uint8


def test_preprocessor_inference_pipeline():
    prep = MedicalImagePreprocessor(image_size=(224, 224))
    img = Image.fromarray(np.random.randint(0, 255, (250, 250), dtype=np.uint8))

    tensor, display = prep.preprocess_for_inference(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert display.shape == (224, 224, 3)
    assert 0.0 <= display.max() <= 1.0


def test_preprocessor_rejects_empty():
    prep = MedicalImagePreprocessor()
    with pytest.raises(ValueError):
        prep.validate_image(np.array([]))
