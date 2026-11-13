"""ResNet-50 Architecture for Multi-Label Chest Radiograph Classification."""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50XRay(nn.Module):
    """Pretrained ResNet-50 transfer learning model."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Retain conv stem and residual stages 1-4
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool
        num_features = backbone.fc.in_features  # 2048

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def get_target_layer(self) -> nn.Module:
        """Returns final residual block (layer4) for Grad-CAM."""
        return self.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
