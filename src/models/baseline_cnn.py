"""Baseline Custom CNN for Chest Radiograph Multi-Label Classification."""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """A 4-stage convolutional network establishing the experimental baseline.

    Features:
    - 4 Conv2d blocks with BatchNorm, LeakyReLU, and MaxPool2d
    - Spatial feature maps reduced from 224x224 to 14x14
    - Global Average Pooling (GAP) to prevent overfitting
    - Multi-label classification head outputting 14 unnormalized logits
    """

    def __init__(self, num_classes: int = 14, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor
        self.features = nn.Sequential(
            # Stage 1: (3, 224, 224) -> (32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 2: (32, 112, 112) -> (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 3: (64, 56, 56) -> (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 4: (128, 28, 28) -> (256, 14, 14)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, -3.5)

    def get_target_layer(self) -> nn.Module:
        """Returns the final convolutional block for Grad-CAM."""
        return self.features[12]  # Conv2d(128, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.global_pool(feat)
        flat = torch.flatten(pooled, 1)
        dropped = self.dropout(flat)
        logits = self.classifier(dropped)
        return logits
