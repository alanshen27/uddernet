"""Model definitions for UdderNet.

- UdderCNN: custom convolutional network for binary classification of udder
  images (positive = mastitis, negative = healthy).
- build_resnet18: pretrained ResNet-18 with a fresh 2-class head, for transfer
  learning on small image datasets.
- MastitisMLP: fully-connected network for the tabular milk-sensor dataset.
"""

import torch
from torch import nn
from torchvision import models as tv_models

# Recommended input resolution per architecture.
IMG_SIZE = {"cnn": 128, "resnet18": 224}


class UdderCNN(nn.Module):
    """Simple CNN for 128x128 RGB images, binary output (2 logits)."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),    # 128 -> 64
            conv_block(32, 64),   # 64 -> 32
            conv_block(64, 128),  # 32 -> 16
            conv_block(128, 256), # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_resnet18(num_classes: int = 2, pretrained: bool = True,
                   freeze_backbone: bool = False) -> nn.Module:
    """ResNet-18 with its final layer replaced by a `num_classes`-way head.

    With pretrained=True the ImageNet backbone is reused (transfer learning),
    which generalises far better than training from scratch on a few hundred
    images. Set freeze_backbone=True to train only the new head.
    """
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Factory used by both training and inference so checkpoints stay portable."""
    if name == "cnn":
        return UdderCNN(num_classes=num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model '{name}'. Choose from: cnn, resnet18.")


class MastitisMLP(nn.Module):
    """MLP for tabular milk-sensor features, binary output (2 logits)."""

    def __init__(self, in_features: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
