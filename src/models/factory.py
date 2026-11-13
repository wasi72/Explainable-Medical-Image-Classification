"""Factory pattern for model instantiation, parameter freezing, and layer hooks."""

import os
from typing import Optional
import torch
import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN
from src.models.densenet import DenseNet121XRay
from src.models.resnet import ResNet50XRay
from src.models.efficientnet import EfficientNetXRay
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Instantiates models and handles fine-tuning / freeze schedules."""

    SUPPORTED_MODELS = ["densenet121", "resnet50", "efficientnet_b0", "baseline_cnn"]

    @classmethod
    def create_model(
        cls,
        model_name: str = "densenet121",
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        weights_path: Optional[str] = None
    ) -> nn.Module:
        """Instantiates the specified neural network architecture.

        Args:
            model_name: One of ['densenet121', 'resnet50', 'efficientnet_b0', 'baseline_cnn'].
            num_classes: Number of disease targets (14 for NIH ChestX-ray14).
            pretrained: Whether to load ImageNet pretrained weights.
            dropout_rate: Dropout probability in classification head.
            freeze_backbone: Whether to freeze feature extractor layers for Stage 1 warm-up.
            weights_path: Optional path to custom trained checkpoint (.pth).

        Returns:
            Configured PyTorch nn.Module.
        """
        model_name = model_name.lower().strip()
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {cls.SUPPORTED_MODELS}"
            )

        logger.info(f"Instantiating model '{model_name}' (pretrained={pretrained}, classes={num_classes})")

        if model_name == "densenet121":
            model = DenseNet121XRay(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
        elif model_name == "resnet50":
            model = ResNet50XRay(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
        elif model_name == "efficientnet_b0":
            model = EfficientNetXRay(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
        elif model_name == "baseline_cnn":
            model = BaselineCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unhandled model name: {model_name}")

        # Check for trained weights checkpoint
        target_weights = weights_path or os.path.join("weights", f"{model_name}_best.pth")
        if target_weights and os.path.exists(target_weights):
            try:
                ckpt = torch.load(target_weights, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)
                model.load_state_dict(state, strict=False)
                logger.info(f"Successfully loaded trained checkpoint from {target_weights}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint from {target_weights}: {e}")

        if freeze_backbone and model_name != "baseline_cnn":
            cls.freeze_backbone(model, model_name)

        return model

    @staticmethod
    def freeze_backbone(model: nn.Module, model_name: str) -> None:
        """Freezes all layers except the final classification head."""
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier parameters
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        logger.info(f"Frozen backbone parameters for '{model_name}'. Only classifier head is trainable.")

    @staticmethod
    def unfreeze_all(model: nn.Module) -> None:
        """Unfreezes all model parameters for full end-to-end fine-tuning."""
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all model parameters for full fine-tuning.")

    @staticmethod
    def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
        """Retrieves the optimal target convolutional layer for Grad-CAM."""
        if hasattr(model, "get_target_layer"):
            return model.get_target_layer()

        model_name = model_name.lower()
        if "dense" in model_name:
            return model.features.denseblock4
        elif "resnet" in model_name:
            return model.layer4
        elif "efficient" in model_name:
            return model.features[-1]
        elif "baseline" in model_name:
            return model.features[12]
        else:
            raise ValueError(f"Unable to determine target layer for {model_name}")
