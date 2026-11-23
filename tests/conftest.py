import numpy as np
import pytest
from PIL import Image

from geosense import load_manifest


@pytest.fixture(scope="session")
def mini(tmp_path_factory):
    """A tiny copy of the real manifest (4 images per country and split) with synthetic 384x166 images."""
    root = tmp_path_factory.mktemp("compressed_dataset")
    full = load_manifest()
    sub = (full.groupby(["country", "split"], group_keys=False).head(4)).reset_index(drop=True)
    rng = np.random.default_rng(0)
    for rel, lab in zip(sub.path, sub.label):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        base = np.array([(lab * 37) % 256, (lab * 91) % 256, (lab * 53) % 256])
        arr = np.clip(base + rng.integers(-40, 40, (166, 384, 3)), 0, 255).astype("uint8")
        Image.fromarray(arr).save(root / rel)
    return root, sub
