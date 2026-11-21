"""Sanity checks on the shared split. Run with: pytest -q"""
from geosense import load_classes, load_manifest

df = load_manifest()


def test_splits_present_and_sized():
    share = df.split.value_counts(normalize=True)
    assert set(share.index) == {"train", "val", "test"}
    assert abs(share["train"] - 0.70) < 0.02
    assert abs(share["val"] - 0.15) < 0.02
    assert abs(share["test"] - 0.15) < 0.02


def test_no_duplicate_group_leaks_across_splits():
    # near-identical screenshots must never be split between train and test
    assert df.groupby("dup_group").split.nunique().max() == 1


def test_every_image_listed_once():
    assert df.path.is_unique


def test_every_country_in_every_split():
    assert (df.groupby("split").country.nunique() == df.country.nunique()).all()


def test_labels_match_classes_file():
    classes, weights = load_classes()
    assert len(classes) == len(weights) == df.label.nunique()
    assert all(classes[l] == c for l, c in df[["label", "country"]].drop_duplicates().itertuples(index=False))


def test_min_images_per_class():
    assert df.country.value_counts().min() >= 100
