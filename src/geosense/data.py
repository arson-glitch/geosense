"""Load the cleaned GeoSense dataset.

Everyone on the team uses the same manifest (data/manifest.csv), so all models
train, tune and test on exactly the same images.

    from geosense import load_manifest, GeoSenseDataset
    train_ds = GeoSenseDataset("~/Downloads/compressed_dataset", split="train", transform=...)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS = ("train", "val", "test")


def load_manifest(path: str | Path = DATA_DIR / "manifest.csv", split: str | None = None) -> pd.DataFrame:
    """One row per kept image: path (relative to dataset root), country, label, dup_group, split."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_classes(path: str | Path = DATA_DIR / "classes.json") -> tuple[list[str], list[float]]:
    """Class names (index = label) and inverse-frequency class weights from the train split."""
    with open(path) as fh:
        d = json.load(fh)
    return d["classes"], d["class_weights"]


try:
    from PIL import Image
    from torch.utils.data import Dataset

    class GeoSenseDataset(Dataset):
        """PyTorch dataset reading images listed in the manifest from the original Kaggle folder.

        root: the Kaggle `compressed_dataset` folder (one sub-folder per country).
        """

        def __init__(self, root: str | Path, split: str, transform=None,
                     manifest: str | Path = DATA_DIR / "manifest.csv"):
            self.root = Path(root).expanduser()
            if not self.root.is_dir():
                raise FileNotFoundError(f"Dataset folder not found: {self.root}")
            self.df = load_manifest(manifest, split)
            self.transform = transform
            self.classes, _ = load_classes(Path(manifest).with_name("classes.json"))

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, i: int):
            row = self.df.iloc[i]
            img = Image.open(self.root / row["path"]).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, int(row["label"])

except ImportError:  # torch not installed - manifest helpers still work
    pass
