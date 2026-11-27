#!/usr/bin/env bash
# Download everything the GPU jobs need, where there IS internet (usually the login node):
#     bash scripts/slurm/fetch_assets.sh            # dataset + StreetCLIP + checks
#     bash scripts/slurm/fetch_assets.sh --check    # only the checks (e.g. after an rsync)
#
# 1. The Kaggle dataset (6.7 GB) into the parent folder of GEOSENSE_DATA, with kagglehub.
#    Kaggle credentials: kagglehub may ask for your Kaggle API token. Create it on kaggle.com
#    (Settings -> API -> Create New Token) and EITHER put the downloaded kaggle.json in
#    ~/.kaggle/kaggle.json yourself (`chmod 600 ~/.kaggle/kaggle.json`), OR type it into the
#    current shell only:  read -rp "Kaggle user: " KAGGLE_USERNAME; read -rsp "Kaggle key: " KAGGLE_KEY;
#    export KAGGLE_USERNAME KAGGLE_KEY.  This script never writes a credential anywhere.
#    Alternative without Kaggle on the cluster: copy the folder from a laptop that has it:
#      rsync -avP ~/Downloads/geosense_dataset/compressed_dataset/ <user>@<login-host>:<GEOSENSE_DATA>/
#    (trailing slashes matter: they copy the CONTENT of compressed_dataset into GEOSENSE_DATA),
#    then run this script with --check.
# 2. geolocal/StreetCLIP (1.7 GB) into HF_HOME, so the jobs can run with HF_HUB_OFFLINE=1.
# 3. Checks: 49,997 JPEGs, every `path` of data/splits.csv exists, the model loads offline.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
geosense_load_env
geosense_activate

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

if [[ "$CHECK_ONLY" == 0 ]]; then
    # --- 1. dataset -----------------------------------------------------------------------
    if [[ -d "$GEOSENSE_DATA/France" ]]; then
        echo "dataset already at $GEOSENSE_DATA: skipping download"
    else
        parent="$(dirname "$GEOSENSE_DATA")"
        mkdir -p "$parent"
        echo "downloading the Kaggle dataset into $parent/.kagglehub (6.7 GB, takes a while)"
        # kagglehub stores datasets under KAGGLEHUB_CACHE/datasets/<owner>/<slug>/versions/<n>/
        src="$(KAGGLEHUB_CACHE="$parent/.kagglehub" python -c '
import kagglehub
print(kagglehub.dataset_download("ubitquitin/geolocation-geoguessr-images-50k"))' | tail -1)"
        [[ -d "$src/compressed_dataset/France" ]] || geosense_die "unexpected layout in $src"
        if [[ -e "$GEOSENSE_DATA" ]]; then
            geosense_die "$GEOSENSE_DATA exists but has no France/ folder: check it by hand"
        fi
        ln -s "$src/compressed_dataset" "$GEOSENSE_DATA"
        echo "linked $GEOSENSE_DATA -> $src/compressed_dataset"
    fi

    # --- 2. StreetCLIP ----------------------------------------------------------------------
    echo "downloading geolocal/StreetCLIP into HF_HOME=$HF_HOME (1.7 GB, skipped if already cached)"
    mkdir -p "$HF_HOME"
    python - <<'EOF'
from huggingface_hub import snapshot_download
# the two example photos of the model card are not needed
path = snapshot_download("geolocal/StreetCLIP", ignore_patterns=["*.jpg", "*.jpeg"])
print("StreetCLIP cached at", path)
EOF
fi

# --- 3. checks ---------------------------------------------------------------------------
echo
echo "=== checks ==="
n_jpg="$(find -L "$GEOSENSE_DATA" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
echo "JPEG files under $GEOSENSE_DATA: $n_jpg (expected 49997)"
[[ "$n_jpg" == 49997 ]] || echo "WARNING: unexpected number of images (incomplete download or copy?)"

cd "$GEOSENSE_REPO"
HF_HUB_OFFLINE=1 python - <<'EOF'
import os
from pathlib import Path

from geosense import load_splits

root = Path(os.environ["GEOSENSE_DATA"])
df = load_splits()
missing = [p for p in df["path"] if not (root / p).is_file()]
print(f"splits.csv: {len(df)} rows, {df['country'].nunique()} countries, missing files: {len(missing)}")
if missing:
    print("first missing:", missing[:5])
    raise SystemExit("ERROR: some images of splits.csv are missing")

# offline check of the model files (config + tokenizer only: loading the 1.7 GB weights on the
# login node is not needed, the smoke job does it on the GPU node)
from huggingface_hub import try_to_load_from_cache
from transformers import AutoConfig, AutoTokenizer

AutoConfig.from_pretrained("geolocal/StreetCLIP")
AutoTokenizer.from_pretrained("geolocal/StreetCLIP")
weights = try_to_load_from_cache("geolocal/StreetCLIP", "pytorch_model.bin")
if not isinstance(weights, str):
    raise SystemExit("ERROR: StreetCLIP weights not in HF_HOME (run without --check on a node with internet)")
print(f"StreetCLIP available offline: {weights} ({os.path.getsize(weights) / 1e9:.2f} GB)")
print("OK: data and model ready for HF_HUB_OFFLINE=1 jobs")
EOF
