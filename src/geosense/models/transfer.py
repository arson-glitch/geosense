"""Transfer learning with ImageNet pre-trained models (model_name = "resnet50" / "efficientnet_b0").

Two stages, as seen in class:
  Stage 1 - feature extraction : freeze the whole backbone, train only the new last layer.
  Stage 2 - fine-tuning        : also unfreeze the last block(s), train with a smaller learning rate.

Stage 1 is fast and safe; stage 2 adapts the high-level features (textures, objects)
to street scenes. Early layers (edges, colours) stay frozen - they are generic.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

ARCHS = ("resnet50", "efficientnet_b0")


def build_transfer_model(arch: str, num_classes: int, pretrained: bool = True,
                         dropout: float = 0.3) -> nn.Module:
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_classes))
    else:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    m.arch = arch
    return m


def head(model: nn.Module) -> nn.Module:
    return model.fc if model.arch == "resnet50" else model.classifier


def blocks(model: nn.Module) -> list[nn.Module]:
    """Backbone blocks from first to last (what we unfreeze from the end)."""
    if model.arch == "resnet50":
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    return list(model.features)          # efficientnet: 9 stages


def freeze_backbone(model: nn.Module) -> None:
    """Stage 1: only the new head is trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in head(model).parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, n: int = 1) -> None:
    """Stage 2: also train the last n backbone blocks (resnet50: n=1 -> layer4)."""
    for b in blocks(model)[-n:]:
        for p in b.parameters():
            p.requires_grad = True


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Last conv block - the layer Maya should hook for Grad-CAM."""
    return model.layer4[-1] if model.arch == "resnet50" else model.features[-1]
