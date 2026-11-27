"""Write Maya's val figures/tables to results/. Run: python -m geosense.make_results [model ...]

VAL ONLY. Fits one temperature per model on val, then the StreetCLIP + own-model mix weight.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibrate import apply_temperature, fit_mix_weight, fit_temperature, mix_probs
from .data import PROJECT_ROOT, load_classes
from .evaluate import confusion, get_labels, metrics_from_probs, per_country_report, reliability_bins
from .predictions import MODEL_NAMES, PRED_DIR, load_logits

OUT = PROJECT_ROOT / "results"


def main(clip: str = "streetclip_probe", own: str = "resnet50") -> None:
    OUT.mkdir(exist_ok=True)
    classes, _ = load_classes()
    y = get_labels("val")
    names = [m for m in MODEL_NAMES if (PRED_DIR / f"{m}_val_logits.npy").exists()]
    T = {m: fit_temperature(load_logits(m, "val"), y) for m in names}
    probs = {m: apply_temperature(load_logits(m, "val"), T[m]) for m in names}
    w, _ = fit_mix_weight(probs[clip], probs[own], y)
    probs["calibrated_mix"] = mix_probs(probs[clip], probs[own], w)

    table = {m: metrics_from_probs(p, y, classes) for m, p in probs.items()}
    import pandas as pd
    df = pd.DataFrame(table).T
    df.round(4).to_csv(OUT / "val_results_table.csv")
    (OUT / "val_calibration.json").write_text(json.dumps({"temperatures": T, "mix": {"clip": clip, "own": own, "w": w}}, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, m in zip(axes, [own, clip, "calibrated_mix"]):
        for label, p in (("raw", apply_temperature(load_logits(m, "val"), 1.0) if m in names else None),
                         ("calibrated", probs[m])):
            if p is None:
                continue
            b = reliability_bins(p, y)
            ax.plot(b[:, 0], b[:, 1], "o-", label=label)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=m, xlabel="confidence")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "reliability_val.png", dpi=150)

    _, cm_k, kn = confusion(probs["calibrated_mix"], y)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm_k, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(kn)), kn, rotation=45, ha="right")
    ax.set_yticks(range(len(kn)), kn)
    for i in range(len(kn)):
        for j in range(len(kn)):
            ax.text(j, i, f"{cm_k[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set(xlabel="predicted", ylabel="true", title="Continent confusion (val, mix)")
    fig.tight_layout()
    fig.savefig(OUT / "continent_confusion_val.png", dpi=150)

    per_country_report(probs["calibrated_mix"], y, classes).sort_values("recall").to_csv(OUT / "val_per_country.csv")
    print(df.round(3).to_string())
    print("temperatures:", {k: round(v, 2) for k, v in T.items()}, "| mix w =", w)


if __name__ == "__main__":
    main(*sys.argv[1:3])
