"""Load the cleaned GeoSense dataset (team contract: data/splits.csv, classes.json, continents.json).

    from geosense import load_splits, load_manifest, load_classes, crop_ui, GeoSenseDataset
    train_ds = GeoSenseDataset("~/Downloads/compressed_dataset", split="train", transform=...)

Order matters: prediction files (.npy) follow the row order of splits.csv for that split,
and the column order of classes.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS = ("train", "val", "test")

# UI crop (team rule: keep width 5% -> 82% of a 1536 px image, full height).
# The GeoGuessr interface has a fixed size in pixels, so the margins are applied in pixels.
# On the standard 1536x662 images this is exactly 5% -> 82%; on the 1030x703 images
# (0.1% of the data) it also removes the score box and mini-map, which 5% -> 82% would not.
UI_LEFT_PX = round(0.05 * 1536)            # 77 px  (compass + buttons)
UI_RIGHT_PX = 1536 - round(0.82 * 1536)    # 276 px (score box + mini-map)


def crop_ui(img: Image.Image) -> Image.Image:
    """Remove the GeoGuessr interface from the left and right edges."""
    w, h = img.size
    return img.crop((UI_LEFT_PX, 0, w - UI_RIGHT_PX, h))


def load_splits(path: str | Path = DATA_DIR / "splits.csv", split: str | None = None) -> pd.DataFrame:
    """One row per kept image: path (relative to compressed_dataset/), country, continent, split."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"splits file not found: {path}")
    df = pd.read_csv(path)
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_classes(path: str | Path = DATA_DIR / "classes.json",
                 counts_path: str | Path | None = None) -> tuple[list[str], list[float]]:
    """(classes, class_weights).

    classes: kept countries from classes.json, sorted alphabetically - index = class id.
    class_weights: inverse-frequency weights from the train split (class_counts.csv), same order.
    Usage: ``classes, weights = load_classes()``.
    """
    path = Path(path)
    with open(path) as fh:
        classes = json.load(fh)
    counts_path = path.with_name("class_counts.csv") if counts_path is None else Path(counts_path)
    weights = pd.read_csv(counts_path, index_col=0)["class_weight"].reindex(classes).tolist()
    return classes, weights


def load_manifest(path: str | Path = DATA_DIR / "splits.csv", split: str | None = None) -> pd.DataFrame:
    """splits.csv plus a ``label`` column (class id = index in classes.json). Same rows, same order."""
    df = load_splits(path, split)
    classes, _ = load_classes(Path(path).with_name("classes.json"))
    df["label"] = df["country"].map({c: i for i, c in enumerate(classes)})
    return df


def load_continents(path: str | Path = DATA_DIR / "continents.json") -> dict[str, str]:
    """{country: continent} for every kept country."""
    with open(path) as fh:
        return json.load(fh)


try:
    from torch.utils.data import Dataset

    class GeoSenseDataset(Dataset):
        """Images of one split, in splits.csv order, with the UI cropped off.

        root: the Kaggle `compressed_dataset` folder (one sub-folder per country).
        transform: applied after the UI crop (resize short side, crop, ToTensor, ...).
        Returns (image, class_id) with class_id = index in classes.json.
        """

        def __init__(self, root: str | Path, split: str, transform=None, data_dir: str | Path = DATA_DIR):
            self.root = Path(root).expanduser()
            if not self.root.is_dir():
                raise FileNotFoundError(f"Dataset folder not found: {self.root}")
            data_dir = Path(data_dir)
            self.df = load_splits(data_dir / "splits.csv", split)
            self.classes, _ = load_classes(data_dir / "classes.json")
            self.class_to_id = {c: i for i, c in enumerate(self.classes)}
            self.labels = self.df["country"].map(self.class_to_id).to_numpy()
            self.transform = transform

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, i: int):
            img = crop_ui(Image.open(self.root / self.df["path"].iloc[i]).convert("RGB"))
            if self.transform is not None:
                img = self.transform(img)
            return img, int(self.labels[i])

except ImportError:  # torch not installed - the helpers above still work
    pass
