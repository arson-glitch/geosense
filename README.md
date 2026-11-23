# GeoSense – Where was this photo taken?

Predict the **country** an outdoor photo was taken in, from GeoGuessr / Street View screenshots.
Group project · AI – Machine Learning & Analytics · Prof. Paula Delgado-Santos · IE University

**Team:** Louis Levet-Lecarpentier (coordination, business case, prototype) · Seif Omara (data cleaning, EDA) ·
Thea Taleb (baselines) · Mohamed Embaby (CNN from scratch) · Maya Tamimi (transfer learning) ·
Faris Alami (evaluation, Grad-CAM, responsible AI)

## Setup

```bash
conda env create -f environment.yml
conda activate geosense
python -m pip install -e .
pytest -q            # checks the shared split is consistent
```

## Data

The images are **not** in this repo (6.8 GB, and Street View imagery may not be redistributed).
Each team member downloads them once:

1. Download [GeoLocation – Geoguessr Images (50K)](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k) from Kaggle and unzip it.
   You get a folder `compressed_dataset/` with one sub-folder per country.
2. Keep it **outside** this repo, e.g. `~/Downloads/compressed_dataset`.

The repo contains the **cleaned, shared split** in `data/`:

| File | What it is |
|---|---|
| `manifest.csv` | Every usable image: `path, country, label, dup_group, split`. **Everyone trains/tests on this.** |
| `classes.json` | Country names (list index = label) and class weights from the train split |
| `class_counts.csv` | Images per country per split |
| `cleaning_report.json` | What was removed and why, all parameters |
| `image_stats.csv` | Per-image file size and colour statistics, used by the EDA notebook (created by `geosense.clean`) |

### Loading the data

**Option A – straight from the manifest (recommended, no extra files):**

```python
from geosense import GeoSenseDataset
from torchvision import transforms

tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_ds = GeoSenseDataset("~/Downloads/compressed_dataset", split="train", transform=tf)
val_ds   = GeoSenseDataset("~/Downloads/compressed_dataset", split="val",   transform=tf)
```

**Option B – train/val/test folders for `torchvision.datasets.ImageFolder`:**

```bash
python -m geosense.split --root ~/Downloads/compressed_dataset --out ~/Downloads/geosense_split
```

Uses hard links (no extra disk space). Gives `geosense_split/{train,val,test}/<country>/*.jpg`.

> ⚠️ Do not create your own split. Use `split` from the manifest so all results are comparable.
> The **test** split is used only once, at the very end.

## Data cleaning (`src/geosense/clean.py`)

Non-destructive: images are never modified, the output is the manifest.

| Step | Result |
|---|---|
| Raw data | 49,997 images, 124 countries, all 1536×662 px JPEG |
| Corrupt / tiny / blank files | 0 found |
| Exact duplicates (identical bytes) | 1 removed |
| Near-duplicates (same screenshot, different round/score overlay; pHash distance 0) | 385 images in 194 groups – kept, but each group is always in **one** split, so none leaks into test |
| Countries with < 100 images | 68 countries, 2,006 images removed |
| **Final** | **47,990 images, 56 countries** |
| Split (stratified by country, grouped by duplicate group, seed 42) | train 33,589 (70%) · val 7,193 (15%) · test 7,208 (15%) |

Decisions and evidence:

- **Duplicate threshold = 0.** At pHash distance 0, 100% of matched pairs were the same country and visually the
  same location. At distance 2 only 80% were, at 4 only 30%: the matches were different places with similar layouts
  (e.g. flat horizons in Canada and the USA). A looser threshold chained 273 unrelated images into one group.
  Cross-country matches are never treated as duplicates.
- **Cut-off = 100 images per country** (proposal's plan). Alternatives: ≥150 → 44 countries / 93% of images;
  ≥200 → 40 / 92%. The smallest test classes have ~15 images, so per-country test scores for small countries are noisy.
- **Imbalance:** United States = 12,014 images (25%). The weights in `classes.json` are plain inverse frequency
  (Ghana ≈ 110× USA); consider square-root weights or a weighted sampler if training is unstable.

Known limitation: hashing only catches copies of the *same* screenshot. Photos of the same place facing another
direction look different and may still sit in different splits.

Re-run (≈3–5 min, then seconds for parameter changes):

```bash
python -m geosense.clean --root ~/Downloads/compressed_dataset --out data/ --min-images 100 --dup-threshold 0
```

## Project structure

```
geosense/
├── README.md
├── pyproject.toml
├── environment.yml
├── data/            manifest + class info (no images)
├── notebooks/       01_eda.ipynb, then baselines, models
├── src/geosense/    clean.py, split.py, data.py
├── results/         figures, tables, summary.md
└── tests/           test_manifest.py
```


## Baselines and StreetCLIP without training (Thea)

Models 1, 2, 5 and 6 of the results table. Code in `src/geosense/` (`preprocess.py`, `baselines.py`, `streetclip.py`, `predictions.py`), notebooks `02_baselines_thea.ipynb` (CPU) and `03_streetclip_thea.ipynb` (GPU).

| Output | Content |
|---|---|
| `predictions/<model>_<split>_logits.npy` | `float32`, shape `(N, 56)`. Rows = `manifest.csv` rows of that split **in file order**, columns = `classes.json` order. Logits, not probabilities (`log(p + 1e-9)` for RF / LR). Models: `majority`, `histogram_lr`, `histogram_rf`, `streetclip_zeroshot`, `streetclip_probe`; splits `val` and `test` |
| `predictions/streetclip_embeddings_<split>.npy` | 768-d unit-norm StreetCLIP image embeddings, **float16** (`.astype("float32")` after loading), same row order. Too big for git: shared by link |

Shared pre-processing (`geosense.preprocess`): keep x from 5% to 82% of the width (removes score box, minimap, compass), resize the short side then centre-crop (no squash unless compared on `val`), 336 px for StreetCLIP, no horizontal flip.
Check your own exports with `from geosense.predictions import check_val; check_val()` (validation metrics only).

## Training our own models: CNN from scratch + ResNet-50 (Mohamed)

CNN from scratch and ResNet-50 / EfficientNet-B0 share one training loop. On a GPU use
`notebooks/04_train_own_models_colab.ipynb` (checkpoints saved to Google Drive).

```bash
# quick check, runs on a laptop CPU in a few minutes (no export)
python -m geosense.train --root ~/Downloads/compressed_dataset --model cnn_scratch --subset 1000 --epochs 2

# full runs (GPU)
python -m geosense.train --root /path/compressed_dataset --model cnn_scratch
python -m geosense.train --root /path/compressed_dataset --model resnet50 --stage1-epochs 3 --unfreeze-blocks 1
# disconnected? same command + --resume
# on Colab add --cache /content/cache: images pre-resized once, epochs ~4x faster
```

| Step | Choice |
|---|---|
| Pre-processing (`transforms.py`) | Same UI crop as `geosense.preprocess` (keep 5–82% of width) → resize short side + crop (`--resize crop`, team rule; `squash` / `wide` only to compare on val) → ImageNet normalisation. Images decoded at half resolution (768×331) for speed |
| Augmentation (train only) | Random crop after resizing to 115%, brightness/contrast/saturation jitter (no hue: soil and vegetation colour are cues). **No horizontal flip** (driving side) |
| Imbalance | Cross-entropy with square-root class weights (`--imbalance sqrt_weights`); `weights`, `sampler`, `none` also available |
| Optimiser | Adam, LR halved when val macro-F1 plateaus, early stopping on val macro-F1 (patience 4) |
| Transfer learning | Stage 1: frozen backbone, new head, lr 1e-3. Stage 2: last block unfrozen, lr 1e-4 |
| Outputs | `checkpoints/<model>/best.pt` (git-ignored), `results/<model>_history.csv`, `predictions/<model>_<split>_logits.npy` |

Exported logits follow the shared contract: `float32`, shape `(N, C)`, rows in `manifest.csv` order for that
split, columns in `classes.json` order, raw logits (before softmax). Only `val` is exported by default;
test logits are exported once at the end with `--export-only --export val,test`. Exports go through `geosense.predictions.save_logits`, so `check_val()` works on them.
For Grad-CAM, hook `model.gradcam_layer()` (CNN) or `geosense.models.gradcam_layer(model)` (ResNet / EfficientNet).

## Data licence

Images derive from Google Street View via GeoGuessr. Used strictly for non-commercial academic purposes.
