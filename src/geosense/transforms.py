"""Image pre-processing for our own models (CNN from scratch, ResNet / EfficientNet).

Pipeline for one image (1536x662 GeoGuessr screenshot):
  1. CropUI      : keep x from 5% to 82% of the width (removes compass/buttons on the
                   left, score box and mini-map on the right). Full height is kept.
  2. Resize      : RESIZE MODE, one switch. Default "crop" = the team rule (same as geosense.preprocess):
                     "crop"   -> short side = size, then crop a square (no distortion, loses the sides)
                     "squash" -> stretch to size x size   (keeps the whole scene, distorts shapes)
                     "wide"   -> size x 2*size            (keeps scene and shape; CNNs accept it)
                   Only switch away from "crop" after comparing on the validation set.
  3. Train only  : random crop (small zoom) + colour jitter. NO horizontal flip:
                   flipping swaps the driving side, which is one of the strongest cues.
  4. ToTensor + Normalize with ImageNet statistics (needed for pre-trained models,
                   harmless for the CNN from scratch, and keeps all our models identical).

StreetCLIP does NOT use this file: Thea's geosense.preprocess handles it at 336 px.
The UI crop itself is shared with her code (crop_overlay), so all models see the same pixels.
"""
from __future__ import annotations

from torchvision import transforms as T

from .preprocess import LEFT as UI_LEFT, RIGHT as UI_RIGHT, crop_overlay

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RESIZE_MODES = ("crop", "squash", "wide")
DEFAULT_RESIZE = "crop"                  # team rule: resize short side, then crop (no squash)


class CropUI:
    """Cut the GeoGuessr interface off the left and right edges (works on PIL images)."""

    def __init__(self, left: float = UI_LEFT, right: float = UI_RIGHT):
        assert 0 <= left < right <= 1
        self.left, self.right = left, right

    def __call__(self, img):
        return crop_overlay(img, self.left, self.right)

    def __repr__(self):
        return f"CropUI(left={self.left}, right={self.right})"


def open_image(path):
    """Open a screenshot as RGB, decoded at half resolution (768x331).

    JPEG can decode at 1/2 scale almost for free, and 331 px of height is still more than
    the 224-258 px we resize to, so nothing is lost - but loading is ~3x faster, which
    matters on Colab where the CPU (not the GPU) is the bottleneck.
    Use this everywhere our models see an image (training, export, the app).
    """
    from PIL import Image
    img = Image.open(path)
    img.draft("RGB", (img.width // 2, img.height // 2))
    return img.convert("RGB")


def output_size(size: int = 224, resize: str = DEFAULT_RESIZE) -> tuple[int, int]:
    """(height, width) of the tensor the model receives."""
    if resize not in RESIZE_MODES:
        raise ValueError(f"resize must be one of {RESIZE_MODES}, got {resize!r}")
    return (size, 2 * size) if resize == "wide" else (size, size)


def build_transforms(train: bool, size: int = 224, resize: str = DEFAULT_RESIZE,
                     zoom: float = 1.15, jitter: float = 0.2):
    """Return the torchvision pipeline.

    train : True -> with augmentation (train split only); False -> deterministic (val / test / app).
    zoom  : for training we resize to `zoom * size` and take a random crop of `size`,
            i.e. a small random shift/zoom. For evaluation we resize straight to `size`.
    """
    h, w = output_size(size, resize)
    steps = [CropUI()]

    if train:
        big = (round(h * zoom), round(w * zoom))
        if resize == "crop":
            steps += [T.Resize(big[0]), T.RandomCrop((h, w))]   # random horizontal position too
        else:                                                   # squash / wide
            steps += [T.Resize(big), T.RandomCrop((h, w))]
        steps += [T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.02)]
        # deliberately NO T.RandomHorizontalFlip()
    else:
        if resize == "crop":
            steps += [T.Resize(h), T.CenterCrop((h, w))]
        else:
            steps += [T.Resize((h, w))]

    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


def denormalize(x):
    """Undo Normalize, for showing a tensor as an image (e.g. Grad-CAM overlays)."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)
