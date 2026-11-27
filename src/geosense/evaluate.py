"""Evaluation module (Maya): all team metrics from the exported .npy logits.

    from geosense.evaluate import evaluate_model, results_table
    evaluate_model("resnet50", "val")

Metrics: top-1, top-5, macro-F1, continent accuracy, distance error (km, between country
centroids), confusion matrices (country + continent), reliability diagram + ECE.

TEST RULE: the test split is used once, at the very end. Asking for split="test" raises
unless you pass final=True on purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from .calibrate import apply_temperature, continent_index, continent_probs
from .data import load_classes, load_continents, load_manifest
from .geo import distance_matrix
from .predictions import load_logits, softmax


def get_labels(split: str, final: bool = False) -> np.ndarray:
    if split == "test" and not final:
        raise PermissionError("The test set is used once, at the very end. Pass final=True to confirm.")
    return load_manifest(split=split)["label"].to_numpy()


def top_k_accuracy(p: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    topk = np.argpartition(-p, k - 1, axis=1)[:, :k]
    return float((topk == y[:, None]).any(1).mean())


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on the top-1 confidence."""
    conf, correct = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """(mean confidence, accuracy, count) per bin, empty bins dropped."""
    conf, correct = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return np.array(rows)


def metrics_from_probs(p: np.ndarray, y: np.ndarray, classes=None, continents=None) -> dict:
    """All headline metrics for one probability matrix (N, C) and labels y."""
    classes = load_classes()[0] if classes is None else classes
    continents = load_continents() if continents is None else continents
    pred = p.argmax(1)
    names, cid = continent_index(classes, continents)
    dist = distance_matrix(classes)[y, pred]
    return {
        "top1": float((pred == y).mean()),
        "top5": top_k_accuracy(p, y, 5),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y))),
        "continent_acc": float((cid[pred] == cid[y]).mean()),
        "dist_km_mean": float(dist.mean()),
        "dist_km_median": float(np.median(dist)),
        "nll": float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()),
        "ece": ece(p, y),
    }


def evaluate_model(model_name: str, split: str = "val", T: float = 1.0, final: bool = False) -> dict:
    y = get_labels(split, final)
    p = apply_temperature(load_logits(model_name, split), T)
    return metrics_from_probs(p, y)


def results_table(model_names, split: str = "val", temps: dict | None = None, final: bool = False) -> pd.DataFrame:
    """One row per model; uses temps[model] if given (fitted on val)."""
    temps = temps or {}
    rows = {m: evaluate_model(m, split, temps.get(m, 1.0), final) for m in model_names}
    return pd.DataFrame(rows).T


def confusion(p: np.ndarray, y: np.ndarray, classes=None, continents=None, normalize: bool = True):
    """(country confusion, continent confusion), rows = true, optionally row-normalised."""
    classes = load_classes()[0] if classes is None else classes
    continents = load_continents() if continents is None else continents
    names, cid = continent_index(classes, continents)
    pred = p.argmax(1)
    norm = "true" if normalize else None
    cm_c = confusion_matrix(y, pred, labels=range(len(classes)), normalize=norm)
    cm_k = confusion_matrix(cid[y], cid[pred], labels=range(len(names)), normalize=norm)
    return np.nan_to_num(cm_c), np.nan_to_num(cm_k), names


def per_country_report(p: np.ndarray, y: np.ndarray, classes=None) -> pd.DataFrame:
    """Per-country support, recall and mean distance error: the base for the bias / Africa analysis."""
    classes = load_classes()[0] if classes is None else classes
    pred = p.argmax(1)
    d = distance_matrix(classes)[y, pred]
    rows = []
    for c in range(len(classes)):
        m = y == c
        if m.any():
            rows.append({"country": classes[c], "n_val": int(m.sum()),
                         "recall": float((pred[m] == c).mean()), "dist_km_mean": float(d[m].mean())})
    return pd.DataFrame(rows).set_index("country")
