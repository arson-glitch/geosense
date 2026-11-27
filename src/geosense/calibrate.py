"""Calibration and model mix (Maya). Everything here is fitted on VAL only.

    T = fit_temperature(val_logits, y_val)          # one number per model
    p = softmax(val_logits / T)
    w = fit_mix_weight(p_clip, p_own, y_val)        # mix = w * p_clip + (1 - w) * p_own
    continent_probs(p, classes, continents)         # sum of the country probabilities
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .predictions import softmax

EPS = 1e-12


def nll(p: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class (the loss temperature scaling minimises)."""
    return float(-np.log(p[np.arange(len(y)), y] + EPS).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising the val NLL of softmax(logits / T). T > 1 = was over-confident."""
    logits = np.asarray(logits, dtype=np.float64)
    res = minimize_scalar(lambda t: nll(softmax(logits / np.exp(t)), y), bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64) / T)


def mix_probs(p_a: np.ndarray, p_b: np.ndarray, w: float) -> np.ndarray:
    """w * p_a + (1 - w) * p_b. Rows still sum to 1."""
    return w * p_a + (1 - w) * p_b


def fit_mix_weight(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray, grid=None) -> tuple[float, float]:
    """Pick w on val by lowest NLL (a proper score: rewards good probabilities, not just top-1).

    Returns (w, nll_at_w). p_a = StreetCLIP, p_b = best own model.
    """
    grid = np.linspace(0, 1, 101) if grid is None else grid
    scores = [nll(mix_probs(p_a, p_b, w), y) for w in grid]
    i = int(np.argmin(scores))
    return float(grid[i]), float(scores[i])


def continent_index(classes: list[str], continents: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """(continent names sorted, array mapping class id -> continent id)."""
    names = sorted(set(continents[c] for c in classes))
    lookup = {n: i for i, n in enumerate(names)}
    return names, np.array([lookup[continents[c]] for c in classes])


def continent_probs(p: np.ndarray, classes: list[str], continents: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """Continent probability = sum of the country probabilities of that continent. Shape (N, n_continents)."""
    names, cid = continent_index(classes, continents)
    out = np.zeros((len(p), len(names)))
    for j in range(len(names)):
        out[:, j] = p[:, cid == j].sum(1)
    return names, out


def cap_confidence(p: np.ndarray, cap: float = 0.99) -> np.ndarray:
    """Display rule: never show 100%. Clip each probability to `cap` for printing only."""
    return np.minimum(p, cap)


def top_k_with_other(p_row: np.ndarray, classes: list[str], k: int = 5) -> list[tuple[str, float]]:
    """[(country, prob) x k] + ('Other', 1 - sum) for the app's top-5 line."""
    idx = np.argsort(-p_row)[:k]
    items = [(classes[i], float(p_row[i])) for i in idx]
    items.append(("Other", float(max(0.0, 1 - sum(v for _, v in items)))))
    return items
