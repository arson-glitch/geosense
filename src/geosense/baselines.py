"""Thea's baselines: majority class and colour-histogram + handmade features (LR / RF).

Features are computed on the CROPPED image (interface removed), split into a top half (sky)
and a bottom half (road / ground). Everything is tuned on the validation set only.
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

from .preprocess import crop_overlay

BINS = 8


def majority_logits(y_train: np.ndarray, n_classes: int, n_rows: int) -> np.ndarray:
    """'Always guess the most frequent country': every row = log of the training class frequencies."""
    prior = np.bincount(y_train, minlength=n_classes) / len(y_train)
    return np.tile(np.log(prior + 1e-9), (n_rows, 1)).astype("float32")


def _features_one(args):
    root, rel = args
    im = Image.open(Path(root) / rel)
    im.draft("RGB", (384, 166))                    # fast, reduced-size JPEG decode
    im = crop_overlay(im.convert("RGB")).resize((128, 64))
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    parts = []
    for region in (rgb[:32], rgb[32:]):            # sky / ground
        for c in range(3):
            parts.append(np.histogram(region[..., c], bins=BINS, range=(0, 1))[0] / region[..., c].size)
        parts += [region.mean((0, 1)), region.std((0, 1))]
        gray = region.mean(2)
        parts.append([np.abs(np.diff(gray, axis=1)).mean(), np.abs(np.diff(gray, axis=0)).mean()])  # edge energy
    parts.append(np.histogram(hsv[..., 0], bins=BINS, range=(0, 1))[0] / hsv[..., 0].size)         # hue
    parts.append([hsv[..., 1].mean(), hsv[..., 2].mean(),
                  ((hsv[..., 0] > 0.20) & (hsv[..., 0] < 0.45) & (hsv[..., 1] > 0.25)).mean(),    # green share
                  ((hsv[..., 0] > 0.50) & (hsv[..., 0] < 0.70) & (hsv[..., 1] > 0.20)).mean()])   # blue share
    return np.concatenate([np.ravel(p) for p in parts]).astype("float32")


def colour_features(root, rel_paths, workers: int = 4) -> np.ndarray:
    """(N, F) hand-made features for the listed images."""
    jobs = [(str(root), p) for p in rel_paths]
    with Pool(workers) as pool:
        return np.stack(pool.map(_features_one, jobs, chunksize=64))


def macro_f1(y, logits) -> float:
    return float(f1_score(y, logits.argmax(1), average="macro"))
