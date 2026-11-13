"""EfficientNet-B0 Architecture for Lightweight Chest Radiograph Inference."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetXRay(nn.Module):
    """EfficientNet-B0 for high-throughput chest radiograph screening."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # In EfficientNet-B0, classifier has 1280 in_features
        num_features = backbone.classifier[1].in_features
        linear = nn.Linear(num_features, num_classes)
        nn.init.normal_(linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(linear.bias, -3.5)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            linear
        )

    def get_target_layer(self) -> nn.Module:
        """Returns final convolutional block for Grad-CAM."""
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
