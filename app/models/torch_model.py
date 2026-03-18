"""
Модель классификации на PyTorch.

Основа: ResNet-18, предобученная на ImageNet (torchvision).
Голова: Dropout → Linear(256) → ReLU → Dropout → Linear(num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ClassificationModel(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18", dropout: float = 0.4):
        super().__init__()
        if backbone == "resnet18":
            base        = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = base.fc.in_features
            base.fc     = nn.Identity()
        elif backbone == "resnet34":
            base        = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            in_features = base.fc.in_features
            base.fc     = nn.Identity()
        elif backbone == "efficientnet_b0":
            base        = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = base
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    # ── Вспомогательные методы тонкой настройки ───────────────────────────────

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_n(self, n: int = 2):
        """Размораживает последние n слоёв основы для тонкой настройки."""
        layers = list(self.backbone.children())
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


def build_model(num_classes: int, backbone: str = "resnet18") -> ClassificationModel:
    return ClassificationModel(num_classes=num_classes, backbone=backbone)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
