"""Compare models and fine-tuning runs (Faris + Louis, StreetCLIP fine-tuning).

This is the fine-tuning owners' comparison tool, limited to what we need to DECIDE:
which fine-tuning run is best, and whether fine-tuning beats the linear probe by more than noise.
The official evaluation module (all metrics, confusion matrices, reliability diagrams, temperature
scaling) is Maya's. Distance error (km) is left to her module: it needs country centroids we do not have.

Pure numpy / pandas (no torch), works directly on the shared .npy logits:
    summary_table      top-1, top-5, macro-F1, continent accuracy, NLL for several models
    per_group_accuracy top-1 per continent or per country (= recall of that group)
    paired_bootstrap   difference between two models with a 95% interval (same resampled images)
    load_run_histories every results/<run>_history.csv + <run>_config.json in one long table
    run_table / pick_best_run  one row per run; best run on val macro-F1 (subset runs excluded)
    load_available_logits      the .npy files that exist today (missing models are skipped)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------- metrics
def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def top_k(logits: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Share of images whose true class is among the k highest logits."""
    top = np.argsort(-logits, 1)[:, :k]
    return float((top == np.asarray(y)[:, None]).any(1).mean())


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Unweighted mean F1 over the classes present in y or pred (same convention as sklearn)."""
    y, pred = np.asarray(y), np.asarray(pred)
    n = int(max(y.max(), pred.max())) + 1
    tp = np.bincount(y[y == pred], minlength=n)
    support, predicted = np.bincount(y, minlength=n), np.bincount(pred, minlength=n)
    present = (support + predicted) > 0
    f1 = 2 * tp[present] / (support[present] + predicted[present])
    return float(f1.mean())


def nll(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class. Over-confident wrong answers make it large."""
    return float(-log_softmax(logits)[np.arange(len(y)), np.asarray(y)].mean())


def continent_probs(logits: np.ndarray, classes: list[str], continents: dict[str, str]):
    """(N, K) continent probabilities = SUM of the country probabilities (team rule), and the K names."""
    names = sorted(set(continents[c] for c in classes))
    member = np.array([[continents[c] == k for k in names] for c in classes], dtype=np.float64)  # (C, K)
    return np.exp(log_softmax(logits)) @ member, names


def continent_accuracy(logits, y, classes, continents) -> float:
    """The continent can be right even when the country is wrong (e.g. Belgium predicted for France)."""
    p, names = continent_probs(logits, classes, continents)
    true = np.array([names.index(continents[classes[i]]) for i in np.asarray(y)])
    return float((p.argmax(1) == true).mean())


METRICS = {
    "top1": lambda logits, y: top_k(logits, y, 1),
    "top5": lambda logits, y: top_k(logits, y, 5),
    "macro_f1": lambda logits, y: macro_f1(y, np.asarray(logits).argmax(1)),
    "nll": nll,
}


def summary_table(models: dict[str, np.ndarray], y, classes, continents) -> pd.DataFrame:
    """One row per model: top1, top5, macro_f1, continent_acc, nll, and mean confidence (max probability).

    mean_conf much higher than top1 = over-confident (common after fine-tuning) -> temperature scaling.
    """
    rows = {}
    for name, logits in models.items():
        p_max = np.exp(log_softmax(logits)).max(1)
        rows[name] = {"top1": top_k(logits, y, 1), "top5": top_k(logits, y, 5),
                      "macro_f1": macro_f1(y, logits.argmax(1)),
                      "continent_acc": continent_accuracy(logits, y, classes, continents),
                      "nll": nll(logits, y), "mean_conf": float(p_max.mean())}
    return pd.DataFrame.from_dict(rows, orient="index")


def per_group_accuracy(logits: np.ndarray, y, groups) -> pd.DataFrame:
    """Top-1 accuracy inside each group of TRUE labels (groups[i] = continent or country of image i).

    Per country this is the recall of that country. Returns columns n, top1, sorted by group name.
    """
    df = pd.DataFrame({"group": np.asarray(groups), "correct": np.asarray(logits).argmax(1) == np.asarray(y)})
    return df.groupby("group")["correct"].agg(n="size", top1="mean")


def paired_bootstrap(logits_a, logits_b, y, metric="macro_f1", n_boot: int = 1000, seed: int = 0) -> dict:
    """metric(A) - metric(B) with a 95% percentile interval.

    Both models are scored on the SAME resampled images each time (paired), so the interval reflects
    only the noise of having 7,171 val images (some countries have ~15). If the interval contains 0,
    we cannot say A is better than B. metric: a name in METRICS or a function (logits, y) -> float.
    """
    f = METRICS[metric] if isinstance(metric, str) else metric
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[b] = f(logits_a[idx], y[idx]) - f(logits_b[idx], y[idx])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"metric": metric if isinstance(metric, str) else getattr(f, "__name__", "metric"),
            "a": f(logits_a, y), "b": f(logits_b, y), "diff": f(logits_a, y) - f(logits_b, y),
            "low": float(low), "high": float(high), "significant": bool(low > 0 or high < 0)}


# ------------------------------------------------------------------------- fine-tuning runs
HISTORY_COLUMNS = ("epoch", "train_loss", "val_loss", "val_acc", "val_top5", "val_macro_f1", "lr", "seconds")
CONFIG_COLUMNS = ("method", "lr", "head_lr", "unfreeze_blocks", "lora_r", "imbalance", "label_smoothing",
                  "epochs", "batch_size", "accum", "trainable_params", "total_params")


def _read_config(path: Path) -> dict:
    """Flat dict of the config fields we compare on; tolerant of a nested {"args": {...}} layout."""
    if not path.exists():
        return {}
    cfg = json.loads(path.read_text())
    flat = {**cfg.get("args", {}), **{k: v for k, v in cfg.items() if not isinstance(v, dict)}}
    flat = {k.replace("-", "_"): v for k, v in flat.items()}
    return {k: flat[k] for k in CONFIG_COLUMNS if k in flat}


def load_run_histories(results_dir) -> pd.DataFrame:
    """One long table (one row per run and epoch) from every <run>_history.csv in results_dir.

    Adds the run's config (<run>_config.json; missing -> has_config False, config columns empty) and
    flags debug runs whose name ends in _subset<N>. Files without epoch / val_macro_f1 (e.g. another
    team member's training loop) are skipped. Empty DataFrame if there are no runs yet.
    """
    frames = []
    for f in sorted(Path(results_dir).glob("*_history.csv")):
        run = f.name[: -len("_history.csv")]
        h = pd.read_csv(f)
        if not {"epoch", "val_macro_f1"} <= set(h.columns):
            continue
        cfg_path = f.with_name(f"{run}_config.json")
        cfg = _read_config(cfg_path)
        m = re.search(r"_subset(\d+)$", run)
        h.insert(0, "run", run)
        h["subset"] = int(m.group(1)) if m else 0
        h["has_config"] = cfg_path.exists()
        for k in CONFIG_COLUMNS:
            h[k] = cfg.get(k, np.nan)
        frames.append(h)
    if not frames:
        return pd.DataFrame(columns=["run", *HISTORY_COLUMNS, "subset", "has_config", *CONFIG_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def run_table(histories: pd.DataFrame) -> pd.DataFrame:
    """One row per run: its best epoch on val macro-F1 (= the epoch saved in best.pt), next to the epoch-0 score.

    best_epoch 0 means training never beat the untrained (zero-shot) model: best.pt is then just zero-shot.
    """
    rows = []
    for run, h in histories.groupby("run", sort=False):
        trained = h[h.epoch >= 1]
        zero = h.loc[h.epoch == 0, "val_macro_f1"]
        best = h.loc[h.val_macro_f1.idxmax()] if len(h) else None   # first max, like finetune.py's strict ">"
        rows.append({
            "run": run, "method": h["method"].iloc[0], "subset": int(h["subset"].iloc[0]),
            "epochs_done": int(trained.epoch.max()) if len(trained) else 0,
            "best_epoch": int(best.epoch) if best is not None else np.nan,
            "best_val_macro_f1": float(best.val_macro_f1) if best is not None else np.nan,
            "val_top1_at_best": float(best.val_acc) if best is not None and "val_acc" in h else np.nan,
            "zero_shot_macro_f1": float(zero.iloc[0]) if len(zero) else np.nan,
            "trainable_params": h["trainable_params"].iloc[0],
            "sec_per_epoch": float(trained.seconds.mean()) if len(trained) and "seconds" in h else np.nan,
        })
    cols = ["run", "method", "subset", "epochs_done", "best_epoch", "best_val_macro_f1", "val_top1_at_best",
            "zero_shot_macro_f1", "trainable_params", "sec_per_epoch"]
    return pd.DataFrame(rows, columns=cols).set_index("run")


def pick_best_run(histories: pd.DataFrame) -> str | None:
    """Name of the run with the highest val macro-F1 at its best epoch, subset runs excluded. None if none."""
    t = run_table(histories)
    t = t[(t.subset == 0) & t.best_val_macro_f1.notna()]
    return None if t.empty else str(t.best_val_macro_f1.idxmax())


# ------------------------------------------------------------------------------ predictions
def load_available_logits(pred_dir, names, split: str = "val", n_rows: int | None = None) -> dict[str, np.ndarray]:
    """{name: logits} for every <name>_<split>_logits.npy that exists (missing models are just skipped).

    n_rows: expected number of rows; files with another shape (stale or subset exports) are skipped.
    """
    out = {}
    for name in names:
        f = Path(pred_dir) / f"{name}_{split}_logits.npy"
        if f.exists():
            x = np.load(f)
            if n_rows is None or x.shape[0] == n_rows:
                out[name] = x.astype(np.float32)
    return out
