"""Build train/ val/ test/ folders (one sub-folder per country) from the manifest.

Only needed if you prefer torchvision.datasets.ImageFolder over GeoSenseDataset.
Uses hard links by default: no extra disk space, files behave like normal copies.

    python -m geosense.split --root ~/Downloads/compressed_dataset --out ~/Downloads/geosense_split
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .data import DATA_DIR, SPLITS, load_manifest


def build_split_folders(root: str | Path, out: str | Path, manifest=DATA_DIR / "manifest.csv",
                        copy: bool = False) -> dict[str, int]:
    root, out = Path(root).expanduser(), Path(out).expanduser()
    if out.resolve().is_relative_to(root.resolve()):
        raise ValueError("Put the output folder OUTSIDE the dataset folder, "
                         "otherwise it would be read as an extra country.")
    df = load_manifest(manifest)
    made = 0
    for path, country, split in df[["path", "country", "split"]].itertuples(index=False):
        dst = out / split / country / Path(path).name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = root / path
        if copy:
            shutil.copy2(src, dst)
        else:
            try:
                os.link(src, dst)
            except OSError:  # different disk / filesystem without hard links
                shutil.copy2(src, dst)
        made += 1
    counts = {s: sum(1 for _ in (out / s).rglob("*.jpg")) for s in SPLITS}
    for s in SPLITS:
        expected = int((df.split == s).sum())
        assert counts[s] == expected, f"{s}: {counts[s]} files, expected {expected}"
    print(f"created {made} files -> {out}  {counts}")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Kaggle compressed_dataset folder")
    ap.add_argument("--out", required=True, help="output folder (outside the dataset folder)")
    ap.add_argument("--copy", action="store_true", help="real copies instead of hard links (+6.6 GB)")
    a = ap.parse_args()
    build_split_folders(a.root, a.out, copy=a.copy)


if __name__ == "__main__":
    main()
