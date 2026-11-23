"""Pre-resized image cache: makes every training epoch ~5x faster on Colab.

Decoding a 1536x662 JPEG is the slowest step of training (the GPU waits for the CPU).
We do it ONCE: for every image in the manifest, remove the GeoGuessr interface (same
crop_overlay as everyone else), resize so the height is `height` px (aspect ratio kept),
and save a small JPEG with the same relative path. Training then reads the small files
and skips the UI crop. Geometry is identical to the uncached pipeline.

    python -m geosense.cache --root /path/compressed_dataset --out /content/cache_258
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

from .data import load_manifest
from .preprocess import crop_overlay
from .transforms import open_image


def cache_height(size: int = 224, zoom: float = 1.15) -> int:
    """Height the training transform resizes to (size * zoom), so the cache loses nothing."""
    return round(size * zoom)


def _one(args):
    root, out, rel, height = args
    dst = Path(out) / rel
    if dst.exists():
        return 0
    img = crop_overlay(open_image(Path(root) / rel))
    w, h = img.size
    img = img.resize((round(w * height / h), height), resample=3)   # 3 = bicubic
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    img.save(tmp, format="JPEG", quality=95)
    os.replace(tmp, dst)                                              # no half-written files
    return 1


def build_cache(root, out, height: int, workers: int | None = None) -> Path:
    root, out = Path(root).expanduser(), Path(out).expanduser()
    paths = load_manifest()["path"].tolist()
    todo = [p for p in paths if not (out / p).exists()]
    if todo:
        print(f"building image cache: {len(todo)} of {len(paths)} images -> {out} (height {height})", flush=True)
        with Pool(workers or os.cpu_count()) as pool:
            for i, _ in enumerate(pool.imap_unordered(_one, [(root, out, p, height) for p in todo], chunksize=64)):
                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=cache_height())
    a = ap.parse_args()
    build_cache(a.root, a.out, a.height)


if __name__ == "__main__":
    main()
