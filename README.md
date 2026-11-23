# GeoSense – Where was this photo taken?

Predict the **country** (and continent) an outdoor street-level photo was taken in, from GeoGuessr / Street View
screenshots. We compare baselines, our own CNNs and StreetCLIP, then calibrate and combine them.
Group project · AI – Machine Learning & Analytics · Prof. Paula Delgado-Santos · IE University

**Team**

| Member | Responsibility |
|---|---|
| Seif Omara | Data, cleaning, de-duplication, EDA |
| Thea Taleb | Baselines + StreetCLIP zero-shot and linear probe |
| Mohamed Embaby | Shared training code, CNN from scratch, ResNet-50 / EfficientNet |
| Faris Alami + Louis Levet-Lecarpentier | StreetCLIP fine-tuning |
| Maya Tamimi | Evaluation, calibration, model mix, explainability, responsible AI |
| Louis Levet-Lecarpentier | Coordination, business case, Streamlit app, report |

## Setup

```bash
conda env create -f environment.yml
conda activate geosense
python -m pip install -e .
pytest -q            # checks the shared data files follow the team contract
```

Intel Mac with PyTorch 2.2: run `pip install "numpy<2"` (PyTorch 2.2 does not work with NumPy 2).

## Data

The images are **not** in this repo (6.7 GB; Street View images, academic non-commercial use only).
Download them once:

```python
import kagglehub
path = kagglehub.dataset_download("ubitquitin/geolocation-geoguessr-images-50k")
# images: <path>/compressed_dataset/<Country>/<file>.jpg
```

or download [the dataset from Kaggle](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)
by hand. Keep the dataset folder **outside** this repo, and don't put other folders inside `compressed_dataset/`.

### Shared files (team contract, do not change without telling the team)

| File | Content |
|---|---|
| `data/splits.csv` | `path, country, continent, split`. `path` is relative to `compressed_dataset/`; `split` is `train` / `val` / `test` |
| `data/classes.json` | Kept countries, sorted alphabetically. **Index in this list = class id everywhere** |
| `data/continents.json` | `{country: continent}` for every kept country |

Prediction files (`predictions/<model>_<split>_logits.npy`) must follow the **row order of `splits.csv`** for that
split and the **column order of `classes.json`**.

Extra files: `class_counts.csv` (images per country, split and continent, plus inverse-frequency class weights),
`cleaning_report.json` (what was removed and why), `image_stats.csv` (per-image size and colour, used by the EDA).

### Loading the data

```python
from geosense import GeoSenseDataset
from torchvision import transforms

tf = transforms.Compose([transforms.Resize(224), transforms.RandomCrop(224),
                         transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor()])   # no horizontal flip!
train_ds = GeoSenseDataset("~/Downloads/compressed_dataset", split="train", transform=tf)
```

`GeoSenseDataset` returns images in `splits.csv` order with the **UI already cropped** (`geosense.crop_ui`) and
labels as class ids from `classes.json`. Use it rather than your own loader so everyone pre-processes the same way.

## Data cleaning (`src/geosense/clean.py`)

Non-destructive: images are never modified; the outputs are the files above.

| Step | Result |
|---|---|
| Raw data | 49,997 images, 124 countries; 49,942 are 1536×662, 55 are 1030×703 |
| Corrupt / tiny / blank files | 0 |
| Duplicates (perceptual hash, distance 0, same country) | 194 removed (1 byte-identical + 193 same screenshot in another game round) |
| Countries with < 100 images | 68 countries, 1,996 images removed |
| **Final** | **47,807 images, 56 countries** |
| Split (stratified by country, seed 42) | train 33,464 · val 7,171 · test 7,172 |

| Continent | Images | Countries |
|---|---|---|
| Europe | 17,538 | 27 |
| North America | 14,253 | 3 |
| Asia | 8,136 | 13 |
| South America | 3,946 | 6 |
| Oceania | 2,254 | 2 |
| Africa | 1,680 | 5 |

Decisions and evidence:

- **Duplicate threshold = 0.** Pairs at distance 0 were 100% the same country and visually the same location.
  At distance 2 only 80% were, at 4 only 30%: different places with similar layouts (e.g. flat horizons).
  Limitation: the same place photographed in another direction is not detected.
- **UI crop in pixels.** The interface has a fixed size in pixels, so `crop_ui` removes 77 px left and 276 px right.
  That is exactly the agreed 5% → 82% on standard images, and it also removes the score box and mini-map on the
  1030×703 images (where 5% → 82% would not). A ~6 px sliver of the score box edge remains top-right (same in every image).
- **Continents:** Russia → Europe, Turkey → Asia, Armenia → Asia, Mexico → North America (to confirm).
- **Imbalance:** United States = 11,973 images (25%). The weights in `class_counts.csv` are plain inverse frequency
  (~110× between the smallest country and the USA); square-root weights or a weighted sampler may train more stably.

Re-run (≈3–5 min):

```bash
python -m geosense.clean --root ~/Downloads/compressed_dataset --out data/
```

## EDA

`notebooks/01_eda.ipynb`; figures and a one-page summary in `results/` (`eda_summary.md`).

## Project structure

```
geosense/
├── README.md
├── pyproject.toml / environment.yml
├── data/            splits.csv, classes.json, continents.json (no images)
├── notebooks/       01_eda.ipynb
├── predictions/     *.npy exports (per model)
├── results/         figures, tables
├── src/geosense/    clean.py, data.py, split.py, ...
└── tests/           test_data_contract.py
```

Never commit images or model weights.

## Baselines and StreetCLIP without training (Thea)

Models 1, 2, 5 and 6 of the results table. Code in `src/geosense/` (`preprocess.py`, `baselines.py`, `streetclip.py`, `predictions.py`), notebooks `02_baselines_thea.ipynb` (CPU) and `03_streetclip_thea.ipynb` (GPU).

| Output | Content |
|---|---|
| `predictions/<model>_<split>_logits.npy` | `float32`, shape `(N, 56)`. Rows = `splits.csv` rows of that split **in file order**, columns = `classes.json` order. Logits, not probabilities (`log(p + 1e-9)` for RF / LR). Models: `majority`, `histogram_lr`, `histogram_rf`, `streetclip_zeroshot`, `streetclip_probe`; splits `val` and `test` |
| `predictions/streetclip_embeddings_<split>.npy` | 768-d unit-norm StreetCLIP image embeddings, **float16** (`.astype("float32")` after loading), same row order. Too big for git: shared by link |

Shared pre-processing (`geosense.preprocess`): keep x from 5% to 82% of the width (removes score box, minimap, compass), resize the short side then centre-crop (no squash unless compared on `val`), 336 px for StreetCLIP, no horizontal flip.
Check your own exports with `from geosense.predictions import check_val; check_val()` (validation metrics only).

## Data licence

## Licence

Street View images (via GeoGuessr): academic non-commercial use only. StreetCLIP: CC-BY-NC.
