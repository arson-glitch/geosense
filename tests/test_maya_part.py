import numpy as np
import pytest

from geosense.calibrate import (apply_temperature, continent_probs, fit_mix_weight, fit_temperature,
                                mix_probs, top_k_with_other)
from geosense.data import load_classes, load_continents
from geosense.evaluate import ece, get_labels, metrics_from_probs, top_k_accuracy
from geosense.geo import CENTROIDS, distance_matrix


def test_centroids_cover_every_class():
    classes, _ = load_classes()
    assert set(classes) == set(CENTROIDS)


def test_distance_matrix_sane():
    classes, _ = load_classes()
    d = distance_matrix(classes)
    assert np.allclose(np.diag(d), 0) and np.allclose(d, d.T)
    i, j = classes.index("France"), classes.index("Belgium")
    assert 200 < d[i, j] < 600


def test_temperature_fixes_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    logits = rng.normal(size=(2000, 5))
    logits[np.arange(2000), y] += 1.0          # weak signal
    logits *= 5                                # far too confident
    T = fit_temperature(logits, y)
    assert T > 2
    p = apply_temperature(logits, T)
    assert ece(p, y) < ece(apply_temperature(logits, 1.0), y)


def test_mix_rows_sum_to_one_and_weight_in_range():
    rng = np.random.default_rng(1)
    a, b = rng.dirichlet(np.ones(6), 50), rng.dirichlet(np.ones(6), 50)
    assert np.allclose(mix_probs(a, b, 0.3).sum(1), 1)
    w, _ = fit_mix_weight(a, b, rng.integers(0, 6, 50))
    assert 0 <= w <= 1


def test_continent_probs_sum_country_probs():
    classes, _ = load_classes()
    conts = load_continents()
    p = np.random.default_rng(2).dirichlet(np.ones(len(classes)), 4)
    names, pc = continent_probs(p, classes, conts)
    assert np.allclose(pc.sum(1), 1)
    j = names.index("Europe")
    assert np.isclose(pc[0, j], sum(p[0, k] for k, c in enumerate(classes) if conts[c] == "Europe"))


def test_top5_with_other_sums_to_one():
    classes, _ = load_classes()
    row = np.random.default_rng(3).dirichlet(np.ones(len(classes)))
    items = top_k_with_other(row, classes)
    assert len(items) == 6 and items[-1][0] == "Other"
    assert np.isclose(sum(v for _, v in items), 1)


def test_metrics_perfect_and_top_k():
    classes, _ = load_classes()
    y = np.arange(len(classes))
    p = np.eye(len(classes))
    m = metrics_from_probs(p, y)
    assert m["top1"] == 1 and m["continent_acc"] == 1 and m["dist_km_mean"] == 0
    assert top_k_accuracy(p, y, 5) == 1


def test_test_split_is_locked():
    with pytest.raises(PermissionError):
        get_labels("test")
