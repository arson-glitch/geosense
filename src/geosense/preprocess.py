"""Shared image pre-processing (team rules from GEOSENSE_CLAUDE_CONTEXT.md, section 4).

1. Crop the GeoGuessr interface: keep x from 5% to 82% of the width, full height.
   (score box + minimap sit at x > 82%, compass and buttons at x < 5%).
2. Never squash by default. Resize the short side, then centre-crop to a square.
   ``mode="squash"`` is available only to compare the two options on the *validation* set.
3. No horizontal flip anywhere (driving side is a clue).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

LEFT, RIGHT = 0.05, 0.82
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def crop_overlay(img: Image.Image, left: float = LEFT, right: float = RIGHT) -> Image.Image:
    """Remove the GeoGuessr interface (left 5% and right 18% of the width)."""
    w, h = img.size
    return img.crop((round(w * left), 0, round(w * right), h))


def to_square(img: Image.Image, size: int, mode: str = "centre") -> Image.Image:
    """Make a size x size image. ``centre``: resize short side then centre-crop. ``squash``: stretch."""
    if mode == "squash":
        return img.resize((size, size), Image.BICUBIC)
    if mode != "centre":
        raise ValueError("mode must be 'centre' or 'squash'")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def load_view(path, size: int, mode: str = "centre", crop: bool = True) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if crop:
        img = crop_overlay(img)
    return to_square(img, size, mode)


class ViewDataset(Dataset):
    """Images from the manifest as normalised tensors (cropped + squared)."""

    def __init__(self, root, rel_paths, size=336, mode="centre", mean=CLIP_MEAN, std=CLIP_STD):
        self.root, self.paths = Path(root).expanduser(), list(rel_paths)
        self.size, self.mode = size, mode
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch

        img = load_view(self.root / self.paths[i], self.size, self.mode)
        a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy((a - self.mean) / self.std), i


def find_dataset_root(extra=()):
    """Locate the Kaggle `compressed_dataset` folder (the one that contains one folder per country)."""
    candidates = [*map(Path, extra), Path("/kaggle/input/geolocation-geoguessr-images-50k/compressed_dataset"),
                  Path("/kaggle/input/geolocation-geoguessr-images-50k"),
                  Path("~/Downloads/compressed_dataset").expanduser(), Path("compressed_dataset")]
    for c in candidates:
        if (c / "France").is_dir():
            return c
        if (c / "compressed_dataset" / "France").is_dir():
            return c / "compressed_dataset"
    kin = Path("/kaggle/input")
    if kin.is_dir():  # newer Kaggle layout: /kaggle/input/datasets/<owner>/<slug>/...
        for hit in sorted(kin.glob("**/France"))[:5]:
            if hit.is_dir():
                return hit.parent
    raise FileNotFoundError("Dataset folder not found. Pass its path (the folder that contains France/, Japan/, ...).")
