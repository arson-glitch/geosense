"""GeoSense - data cleaning.

Non-destructive: images are never modified or deleted. The output is a
manifest (one row per kept image, with its split) that every teammate loads,
so all models train and test on exactly the same data.

Stages
  1. Scan + integrity  : decode every image; record size, mode, md5, pHash, basic colour stats
  2. Bad files         : corrupt / too small / near-blank (loading screens)
  3. Exact duplicates  : identical bytes. Same country -> keep one. Different countries -> drop all (label conflict)
  4. Near duplicates   : pHash Hamming distance <= T -> same dup_group (kept, but never split across train/val/test)
  5. Rare classes      : drop countries with < MIN_IMAGES images after de-duplication
  6. Split             : stratified *group* split 70/15/15 (groups = dup_group), fixed seed

Usage
  python -m geosense.clean --root /path/to/compressed_dataset --out data/
  # stage 1 is slow (decodes all images); re-run with different thresholds instantly:
  python -m geosense.clean --root ... --out data/ --dup-threshold 4 --min-images 150
  # (decoding is resumable: add --max-seconds 150 and re-run until it finishes)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------- stage 1
def scan(root: str) -> pd.DataFrame:
    rows = []
    for country in sorted(os.listdir(root)):
        cdir = os.path.join(root, country)
        if country.startswith((".", "_")) or not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                rows.append({"path": f"{country}/{f}", "country": country})
    return pd.DataFrame(rows)


def inspect_image(args):
    """Decode one image and return integrity info + hash + colour stats."""
    root, rel = args
    import imagehash  # imported in worker

    out = {"path": rel, "ok": False, "error": "", "bytes": 0, "md5": "",
           "width": 0, "height": 0, "mode": "", "phash": "",
           "mean_r": np.nan, "mean_g": np.nan, "mean_b": np.nan,
           "brightness": np.nan, "brightness_std": np.nan}
    try:
        with open(os.path.join(root, rel), "rb") as fh:
            data = fh.read()
        out["bytes"] = len(data)
        out["md5"] = hashlib.md5(data).hexdigest()
        im = Image.open(io.BytesIO(data))
        out["width"], out["height"], out["mode"] = im.width, im.height, im.mode
        # draft = fast reduced-scale JPEG decode; still reads the full stream,
        # so truncated/corrupt files still raise
        im.draft("RGB", (max(1, im.width // 4), max(1, im.height // 4)))
        im = im.convert("RGB")
        im.load()
        out["phash"] = str(imagehash.phash(im))
        a = np.asarray(im, dtype=np.float32)
        out["mean_r"], out["mean_g"], out["mean_b"] = a.reshape(-1, 3).mean(0).round(2)
        gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        out["brightness"], out["brightness_std"] = round(float(gray.mean()), 2), round(float(gray.std()), 2)
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def run_inspection(root: str, df: pd.DataFrame, workers: int, partial_path: str,
                   max_seconds: float | None) -> pd.DataFrame | None:
    """Resumable: results are appended to partial_path in chunks. Returns None if
    stopped by max_seconds before finishing (just run the same command again)."""
    done = set(pd.read_csv(partial_path, usecols=["path"]).path) if os.path.exists(partial_path) else set()
    todo = [p for p in df["path"] if p not in done]
    print(f"  {len(done)} already inspected, {len(todo)} to go", flush=True)
    t0, chunk = time.time(), 2000
    with Pool(workers) as pool:
        for s in range(0, len(todo), chunk):
            batch = pool.map(inspect_image, [(root, p) for p in todo[s:s + chunk]], chunksize=32)
            pd.DataFrame(batch).to_csv(partial_path, mode="a", index=False,
                                       header=not os.path.exists(partial_path))
            el = time.time() - t0
            print(f"  inspected {len(done) + s + len(batch)}/{len(df)}  ({el:.0f}s)", flush=True)
            if max_seconds and el > max_seconds and s + chunk < len(todo):
                print("  time limit reached - re-run the same command to continue", flush=True)
                return None
    res = pd.read_csv(partial_path, keep_default_na=False, na_values=[""]).drop_duplicates("path")
    return df.merge(res, on="path")


# ---------------------------------------------------------------- stage 4 helpers
def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def near_duplicate_pairs(hashes: np.ndarray, threshold: int, hist_max: int = 16,
                         keep_examples_upto: int = 12, block: int = 256):
    """All pairs i<j with Hamming distance <= threshold, plus a distance histogram
    (for choosing the threshold) and a sample of borderline pairs for visual checks."""
    n = len(hashes)
    pairs, examples = [], []
    hist = np.zeros(hist_max + 1, dtype=np.int64)
    for s in range(0, n, block):
        e = min(s + block, n)
        d = popcount64(hashes[s:e, None] ^ hashes[None, :])          # (block, n)
        ii, jj = np.nonzero(d <= max(hist_max, threshold))
        mask = jj > ii + s                                             # i<j only
        ii, jj = ii[mask] + s, jj[mask]
        dd = d[ii - s, jj]
        hist += np.bincount(dd, minlength=hist_max + 1)[: hist_max + 1]
        sel = dd <= threshold
        pairs.append(np.stack([ii[sel], jj[sel], dd[sel]], 1))
        ex = (dd > threshold) & (dd <= keep_examples_upto)
        if ex.any():
            examples.append(np.stack([ii[ex], jj[ex], dd[ex]], 1))
    pairs = np.concatenate(pairs) if pairs else np.zeros((0, 3), int)
    examples = np.concatenate(examples) if examples else np.zeros((0, 3), int)
    return pairs, hist, examples


def union_find_groups(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with one sub-folder per country")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse", action="store_true", help="reuse inventory_raw.csv (skip decoding)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--min-side", type=int, default=100)
    ap.add_argument("--blank-std", type=float, default=8.0, help="grayscale std below this = near-blank image")
    ap.add_argument("--dup-threshold", type=int, default=0, help="pHash Hamming distance for near-duplicates (0 = chosen after visual check: at 2+ most matches are different places with similar layouts)")
    ap.add_argument("--min-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=None, help="stop decoding after N s (resumable)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw_path = os.path.join(a.out, "inventory_raw.csv")
    report: dict = {"params": vars(a)}

    # 1. scan + inspect
    if os.path.exists(raw_path):  # complete inventory exists -> skip decoding
        df = pd.read_csv(raw_path, keep_default_na=False, na_values=[""])
        print(f"[1] reused {len(df)} rows from {raw_path}")
    else:
        df = scan(a.root)
        print(f"[1] found {len(df)} images in {df.country.nunique()} folders; decoding with {a.workers} workers")
        df = run_inspection(a.root, df, a.workers, raw_path + ".partial", a.max_seconds)
        if df is None:
            return 3
        df.to_csv(raw_path, index=False)
    df["ok"] = df["ok"].astype(bool)
    df["status"], df["reason"] = "keep", ""
    report["n_files"], report["n_folders"] = len(df), int(df.country.nunique())
    report["image_sizes"] = df[df.ok].groupby(["width", "height"]).size().sort_values(ascending=False).head(10) \
        .rename("n").reset_index().to_dict("records")

    def drop(mask, reason):
        m = mask & (df.status == "keep")
        df.loc[m, ["status", "reason"]] = ["drop", reason]
        report.setdefault("dropped", {})[reason] = int(m.sum())
        print(f"    drop {reason:<24} {int(m.sum()):>6}")

    # 2. bad files
    print("[2] bad files")
    drop(~df.ok, "corrupt")
    drop(df.ok & (df[["width", "height"]].min(axis=1) < a.min_side), "too_small")
    drop(df.ok & (df.brightness_std < a.blank_std), "near_blank")

    # 3. exact duplicates
    print("[3] exact duplicates (md5)")
    k = df.status == "keep"
    n_labels = df[k].groupby("md5").country.transform("nunique")
    drop(k & (n_labels > 1).reindex(df.index, fill_value=False), "exact_dup_label_conflict")
    k = df.status == "keep"
    drop(k & df[k].duplicated("md5", keep="first").reindex(df.index, fill_value=False), "exact_duplicate")

    # 4. near duplicates -> groups
    print(f"[4] near duplicates (pHash distance <= {a.dup_threshold})")
    kept = df[df.status == "keep"].copy()
    hashes = np.array([int(h, 16) for h in kept.phash], dtype=np.uint64)
    pairs, hist, examples = near_duplicate_pairs(hashes, a.dup_threshold)
    # a "duplicate" across two countries is a hash collision between similar-looking
    # scenes (flat horizons etc.), not the same photo -> never group across countries
    kc = kept.country.to_numpy()
    cross = kc[pairs[:, 0]] != kc[pairs[:, 1]]
    report["near_dup_cross_country_pairs_ignored"] = int(cross.sum())
    print(f"    ignored {int(cross.sum())} cross-country pairs (hash collisions)")
    pairs = pairs[~cross]
    groups = union_find_groups(len(kept), pairs)
    kept["dup_group"] = groups
    gsize = kept.groupby("dup_group").path.transform("size")
    glabels = kept.groupby("dup_group").country.transform("nunique")
    report["near_dup"] = {
        "pairs_within_threshold": int(len(pairs)),
        "pair_distance_histogram": {int(i): int(v) for i, v in enumerate(hist)},
        "images_in_a_group_of_2+": int((gsize > 1).sum()),
        "n_groups_2+": int(kept[gsize > 1].dup_group.nunique()),
        "largest_groups": kept.groupby("dup_group").size().sort_values(ascending=False).head(5).tolist(),
        "mixed_label_groups": int(kept[glabels > 1].dup_group.nunique()),
        "images_in_mixed_label_groups": int((glabels > 1).sum()),
    }
    print(f"    {report['near_dup']['pairs_within_threshold']} pairs, "
          f"{report['near_dup']['images_in_a_group_of_2+']} images in groups, "
          f"largest {report['near_dup']['largest_groups']}, "
          f"mixed-label groups {report['near_dup']['mixed_label_groups']}")
    # save pairs for visual review (both at/below and just above the threshold)
    pp = kept.path.to_numpy()
    pc = kept.country.to_numpy()
    rev = [pd.DataFrame({"a": pp[x[:, 0]], "b": pp[x[:, 1]], "country_a": pc[x[:, 0]],
                         "country_b": pc[x[:, 1]], "distance": x[:, 2]}) for x in (pairs, examples)]
    pd.concat(rev).sort_values("distance").to_csv(os.path.join(a.out, "near_dup_pairs.csv"), index=False)
    df = df.merge(kept[["path", "dup_group"]], on="path", how="left")

    # 5. rare classes
    print(f"[5] classes with < {a.min_images} images")
    k = df.status == "keep"
    counts = df[k].country.value_counts()
    total = int(k.sum())
    report["threshold_table"] = [
        {"min_images": t, "n_classes": int((counts >= t).sum()),
         "pct_images_kept": round(100 * counts[counts >= t].sum() / total, 1)}
        for t in (1, 20, 50, 100, 150, 200, 300, 500)]
    for r in report["threshold_table"]:
        print(f"    >= {r['min_images']:>3}: {r['n_classes']:>3} classes, {r['pct_images_kept']:>5}% of images")
    rare = counts[counts < a.min_images].index
    drop(df.country.isin(rare), "rare_class")

    # 6. split
    print("[6] stratified group split 70/15/15")
    from sklearn.model_selection import StratifiedGroupKFold
    m = df[df.status == "keep"].copy()
    m["dup_group"] = m["dup_group"].astype(int)
    classes = sorted(m.country.unique())
    m["label"] = m.country.map({c: i for i, c in enumerate(classes)})
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=a.seed)
    fold = np.empty(len(m), dtype=int)
    for f, (_, idx) in enumerate(sgkf.split(m, m.label, m.dup_group)):
        fold[idx] = f
    m["split"] = np.select([fold < 3, fold < 6], ["test", "val"], "train")   # 3/20=15%, 3/20=15%
    assert m.groupby("dup_group").split.nunique().max() == 1, "a duplicate group spans splits!"

    per_class = m.pivot_table(index="country", columns="split", values="path", aggfunc="count", fill_value=0)
    per_class = per_class[["train", "val", "test"]]
    train_counts = per_class["train"]
    w = (train_counts.sum() / (len(classes) * train_counts))
    weights = (w / w.mean()).round(4)
    report["final"] = {
        "n_images": len(m), "n_classes": len(classes),
        "split_sizes": m.split.value_counts().to_dict(),
        "smallest_test_class": [str(per_class["test"].idxmin()), int(per_class["test"].min())],
        "largest_class": [str(counts.idxmax()), int(counts.max())],
    }
    print(f"    {report['final']}")

    # outputs
    m[["path", "country", "label", "dup_group", "split"]].sort_values("path") \
        .to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    df.drop(columns=["ok"]).to_csv(os.path.join(a.out, "inventory.csv"), index=False)
    per_class.assign(total=per_class.sum(axis=1), class_weight=weights).sort_values("total", ascending=False) \
        .to_csv(os.path.join(a.out, "class_counts.csv"))
    with open(os.path.join(a.out, "classes.json"), "w") as fh:
        json.dump({"classes": classes, "class_weights": weights.reindex(classes).tolist()}, fh, indent=1)
    with open(os.path.join(a.out, "cleaning_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
