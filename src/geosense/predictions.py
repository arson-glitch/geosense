"""Export / check model predictions in the shared team format.

    predictions/<model_name>_<split>_logits.npy   float32, shape (N, C)
      N = rows of data/manifest.csv with that split, IN THE SAME ORDER AS THE FILE
      C = number of classes, IN THE ORDER OF data/classes.json
Save LOGITS (before softmax). For models that only give probabilities save log(p + 1e-9).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import PROJECT_ROOT, load_classes, load_manifest

PRED_DIR = PROJECT_ROOT / "predictions"
MODEL_NAMES = ("majority", "histogram_lr", "histogram_rf", "cnn_scratch", "resnet50",
               "streetclip_zeroshot", "streetclip_probe", "streetclip_finetuned")


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    """For models that output probabilities (random forest, sklearn LR)."""
    return np.log(p + 1e-9).astype("float32")


def full_class_columns(p: np.ndarray, fitted_classes, n_classes: int) -> np.ndarray:
    """sklearn drops classes absent from training; put every column back in classes.json order."""
    out = np.zeros((len(p), n_classes), dtype=np.float64)
    out[:, np.asarray(fitted_classes, dtype=int)] = p
    return out


def save_logits(model_name: str, split: str, logits: np.ndarray, out_dir: Path = PRED_DIR,
                manifest=None, n_classes: int | None = None) -> Path:
    """Validate shape / dtype / finiteness, then write the .npy file."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}, got {model_name!r}")
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    manifest = load_manifest() if manifest is None else manifest
    n_classes = len(load_classes()[0]) if n_classes is None else n_classes
    n = int((manifest["split"] == split).sum())
    logits = np.asarray(logits, dtype="float32")
    if logits.shape != (n, n_classes):
        raise ValueError(f"{model_name}/{split}: expected shape {(n, n_classes)}, got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError(f"{model_name}/{split}: logits contain NaN or inf")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}_{split}_logits.npy"
    np.save(path, logits)
    return path


def load_logits(model_name: str, split: str, out_dir: Path = PRED_DIR) -> np.ndarray:
    return np.load(Path(out_dir) / f"{model_name}_{split}_logits.npy")


def check_val(out_dir: Path = PRED_DIR, manifest=None) -> None:
    """Print VALIDATION accuracy / macro-F1 for every exported model (never touches test labels)."""
    from sklearn.metrics import f1_score

    manifest = load_manifest() if manifest is None else manifest
    y = manifest.loc[manifest["split"] == "val", "label"].to_numpy()
    print(f"{'model':<24}{'top-1':>8}{'top-5':>8}{'macro-F1':>10}")
    for name in MODEL_NAMES:
        f = Path(out_dir) / f"{name}_val_logits.npy"
        if not f.exists():
            continue
        p = np.load(f)
        if p.shape != (len(y), p.shape[1]) or p.shape[0] != len(y):
            print(f"{name:<24}SKIPPED: {p.shape[0]} rows, expected {len(y)} (stale file from another run?)")
            continue
        top5 = np.argsort(-p, 1)[:, :5]
        print(f"{name:<24}{(p.argmax(1) == y).mean():>8.3f}{np.mean([a in r for a, r in zip(y, top5)]):>8.3f}"
              f"{f1_score(y, p.argmax(1), average='macro'):>10.3f}")
