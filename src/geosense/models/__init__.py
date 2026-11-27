"""Our own image models. StreetCLIP lives elsewhere (Thea / Faris / Louis)."""
from __future__ import annotations

from .cnn import SimpleCNN
from .transfer import ARCHS, build_transfer_model, freeze_backbone, gradcam_layer, unfreeze_last_blocks

MODEL_NAMES = ("cnn_scratch",) + ARCHS


def build_model(name: str, num_classes: int, pretrained: bool = True):
    """name = model_name used in the predictions/ file names."""
    if name == "cnn_scratch":
        return SimpleCNN(num_classes)
    if name in ARCHS:
        return build_transfer_model(name, num_classes, pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


__all__ = ["MODEL_NAMES", "SimpleCNN", "build_model", "build_transfer_model",
           "freeze_backbone", "unfreeze_last_blocks", "gradcam_layer"]
