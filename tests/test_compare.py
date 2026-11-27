"""Offline tests of geosense.compare on small synthetic arrays where the right answer is known."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from geosense import compare

CLASSES = ["Belgium", "France", "Japan", "Kenya"]
CONTINENTS = {"Belgium": "Europe", "France": "Europe", "Japan": "Asia", "Kenya": "Africa"}


def one_hot_logits(pred, n=4, high=10.0):
    z = np.zeros((len(pred), n), dtype=np.float32)
    z[np.arange(len(pred)), pred] = high
    return z


def test_perfect_predictions_score_one_everywhere():
    y = np.array([0, 1, 2, 3, 1, 2])
    t = compare.summary_table({"perfect": one_hot_logits(y, high=50.0)}, y, CLASSES, CONTINENTS)
    for col in ("top1", "top5", "macro_f1", "continent_acc"):
        assert t.loc["perfect", col] == pytest.approx(1.0)
    assert t.loc["perfect", "nll"] == pytest.approx(0.0, abs=1e-6)
    g = compare.per_group_accuracy(one_hot_logits(y), y, [CONTINENTS[CLASSES[i]] for i in y])
    assert (g.top1 == 1.0).all() and g.n.sum() == len(y)


def test_continent_right_when_country_wrong():
    y = np.array([1, 1, 2])                # France, France, Japan
    pred = np.array([0, 0, 2])             # Belgium, Belgium, Japan
    logits = one_hot_logits(pred)
    assert compare.top_k(logits, y, 1) == pytest.approx(1 / 3)
    assert compare.continent_accuracy(logits, y, CLASSES, CONTINENTS) == pytest.approx(1.0)


def test_continent_uses_sum_of_country_probabilities():
    # Kenya is the single most likely country (0.4) but Europe = 0.3 + 0.3 = 0.6 wins as a continent.
    logits = np.log(np.array([[0.3, 0.3, 0.0001, 0.4]]))
    p, names = compare.continent_probs(logits, CLASSES, CONTINENTS)
    assert names[int(p.argmax())] == "Europe"
    assert compare.continent_accuracy(logits, np.array([0]), CLASSES, CONTINENTS) == 1.0
    assert p.sum() == pytest.approx(1.0)


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y, pred = rng.integers(0, 6, 200), rng.integers(0, 5, 200)
    assert compare.macro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="macro"))


def test_bootstrap_model_against_itself_is_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 300)
    logits = rng.normal(size=(300, 4)).astype("float32")
    r = compare.paired_bootstrap(logits, logits, y, "macro_f1", n_boot=200)
    assert r["diff"] == 0 and r["low"] <= 0 <= r["high"] and not r["significant"]


def test_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, 500)
    good, bad = one_hot_logits(y), rng.normal(size=(500, 4))
    r = compare.paired_bootstrap(good, bad, y, "top1", n_boot=200)
    assert r["low"] > 0 and r["significant"]


def write_run(d, run, f1s, config=None):
    epochs = np.arange(len(f1s))
    pd.DataFrame({"epoch": epochs, "train_loss": 1.0, "val_loss": 1.0, "val_acc": f1s,
                  "val_top5": 0.9, "val_macro_f1": f1s, "lr": 1e-5, "seconds": 100.0 * (epochs > 0)}
                 ).to_csv(d / f"{run}_history.csv", index=False)
    if config is not None:
        (d / f"{run}_config.json").write_text(json.dumps(config))


def test_load_histories_subset_and_missing_config(tmp_path):
    write_run(tmp_path, "lb2", [0.50, 0.60, 0.65, 0.63],
              {"method": "last_blocks", "lr": 1e-5, "unfreeze_blocks": 2, "trainable_params": 1000})
    write_run(tmp_path, "lb4_subset200", [0.50, 0.90])                       # debug run, no config
    write_run(tmp_path, "lora8", [0.50, 0.55], {"args": {"method": "lora", "lora_r": 8}})  # nested layout
    write_run(tmp_path, "lb_bad", [0.50, 0.45, 0.40], {"method": "last_blocks", "lr": 3e-5})
    pd.DataFrame({"epoch": [1], "acc": [0.1]}).to_csv(tmp_path / "cnn_scratch_history.csv", index=False)

    h = compare.load_run_histories(tmp_path)
    assert set(h.run) == {"lb2", "lb4_subset200", "lora8", "lb_bad"}         # foreign history file skipped
    sub = h[h.run == "lb4_subset200"]
    assert (sub.subset == 200).all() and not sub.has_config.any() and sub.method.isna().all()
    assert (h.loc[h.run == "lora8", "lora_r"] == 8).all()

    t = compare.run_table(h)
    assert t.loc["lb2", "best_epoch"] == 2 and t.loc["lb2", "best_val_macro_f1"] == pytest.approx(0.65)
    assert t.loc["lb2", "zero_shot_macro_f1"] == pytest.approx(0.50)
    assert t.loc["lb_bad", "best_epoch"] == 0            # never beat zero-shot: best.pt is the epoch-0 model
    assert compare.pick_best_run(h) == "lb2"                                  # 0.90 subset run excluded


def test_no_runs_yet(tmp_path):
    h = compare.load_run_histories(tmp_path)
    assert h.empty and compare.pick_best_run(h) is None


def test_load_available_logits_skips_missing_and_wrong_shape(tmp_path):
    np.save(tmp_path / "a_val_logits.npy", np.zeros((5, 4), "float32"))
    np.save(tmp_path / "b_val_logits.npy", np.zeros((3, 4), "float32"))
    got = compare.load_available_logits(tmp_path, ["a", "b", "missing"], n_rows=5)
    assert list(got) == ["a"]
