"""Image pre-processing for our own models (CNN from scratch, ResNet / EfficientNet).

Pipeline for one image (1536x662 GeoGuessr screenshot):
  1. open_image  : decode at half resolution and remove the GeoGuessr interface with the team
                   crop (geosense.data.crop_ui, margins scaled to the decoded size). Like
                   Seif's GeoSenseDataset, the crop happens when the image is loaded, not here.
  2. Resize      : RESIZE MODE, one switch. Default "crop" = the team rule (same as geosense.preprocess):
                     "crop"   -> short side = size, then crop a square (no distortion, loses the sides)
                     "squash" -> stretch to size x size   (keeps the whole scene, distorts shapes)
                     "wide"   -> size x 2*size            (keeps scene and shape; CNNs accept it)
                   Only switch away from "crop" after comparing on the validation set.
  3. Train only  : random crop (small zoom) + colour jitter (no hue). NO horizontal flip:
                   flipping swaps the driving side, which is one of the strongest cues.
  4. ToTensor + Normalize with ImageNet statistics (needed for pre-trained models,
                   harmless for the CNN from scratch, and keeps all our models identical).

StreetCLIP does NOT use this file: Thea's geosense.preprocess handles it at 336 px.
The UI crop is the shared team function, so all models see the same part of the image.
Usage (training, export, the app):  build_transforms(False)(open_image(path))
"""
from __future__ import annotations

from torchvision import transforms as T

from .data import UI_LEFT_PX, UI_RIGHT_PX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RESIZE_MODES = ("crop", "squash", "wide")
DEFAULT_RESIZE = "crop"                  # team rule: resize short side, then crop (no squash)


def open_image(path, crop_ui: bool = True):
    """Open a screenshot as RGB, decoded at half resolution, with the GeoGuessr interface removed.

    JPEG can decode at 1/2 scale almost for free, and 331 px of height is still more than
    the 224-258 px we resize to, so nothing is lost - but loading is ~3x faster, which
    matters on Colab where the CPU (not the GPU) is the bottleneck.
    The team crop (geosense.data.crop_ui) is defined in pixels of the original image,
    so the margins are scaled by the decoding factor: same region as crop_ui, at any size.
    crop_ui=False for files from geosense.cache (already cropped and resized).
    """
    from PIL import Image
    img = Image.open(path)
    orig_w = img.width
    if img.height >= 600:                        # original screenshots only, never cached small files
        img.draft("RGB", (img.width // 2, img.height // 2))
    img = img.convert("RGB")
    if crop_ui:
        s = img.width / orig_w
        img = img.crop((round(UI_LEFT_PX * s), 0, img.width - round(UI_RIGHT_PX * s), img.height))
    return img


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
    Input: a PIL image with the interface already removed (open_image or GeoSenseDataset).
    """
    h, w = output_size(size, resize)
    steps = []

    if train:
        big = (round(h * zoom), round(w * zoom))
        if resize == "crop":
            steps += [T.Resize(big[0]), T.RandomCrop((h, w))]   # random horizontal position too
        else:                                                   # squash / wide
            steps += [T.Resize(big), T.RandomCrop((h, w))]
        # brightness / contrast / saturation only: NO hue shift, because the colour of soil and
        # vegetation is itself a country cue. Deliberately NO T.RandomHorizontalFlip() either.
        steps += [T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter)]
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
