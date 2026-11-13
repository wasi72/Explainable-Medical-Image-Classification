"""DenseNet-121 Architecture for Chest Radiograph Multi-Label Classification.

Implements the CheXNet-style architecture (Rajpurkar et al., Stanford)
optimized for chest X-ray multi-label pathology detection.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121XRay(nn.Module):
    """DenseNet-121 model with custom multi-label head and Grad-CAM hooks."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load backbone
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)

        # Feature extractor up to final denseblock + norm5
        self.features = backbone.features

        # CheXNet style classifier replacement
        num_features = backbone.classifier.in_features  # 1024
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def get_target_layer(self) -> nn.Module:
        """Returns final convolutional layer inside denseblock4 for Grad-CAM."""
        # The last block in features is norm5 (BatchNorm), right after denseblock4
        # Grad-CAM on denseblock4 provides rich multi-scale spatial activations
        return self.features.denseblock4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits
