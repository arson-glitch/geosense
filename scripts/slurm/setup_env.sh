#!/usr/bin/env bash
# Create (once) the Python environment for GeoSense on the cluster and install the package.
# Run on the LOGIN node (it needs internet for conda/pip):
#     bash scripts/slurm/setup_env.sh
# Idempotent: re-running it skips the env creation and only re-installs / updates the packages.
#
# Why pip for PyTorch: on Linux the PyPI torch wheel ships its own CUDA libraries, so it works on
# the GPU nodes even though this script runs on a login node without a GPU (conda would often
# pick a CPU-only build there).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
geosense_load_env
geosense_load_modules

kind="$(geosense_env_kind)"
echo "environment kind: $kind"

if [[ "$kind" == conda ]]; then
    geosense_init_conda || geosense_die "conda not found (set GS_CONDA_SH or GS_MODULES in cluster.env)"
    if [[ "$GS_CONDA_ENV" == */* ]]; then
        exists=0; [[ -d "$GS_CONDA_ENV/conda-meta" ]] && exists=1
        where=(-p "$GS_CONDA_ENV")
    else
        exists=0; conda env list | awk '{print $1}' | grep -qx "$GS_CONDA_ENV" && exists=1
        where=(-n "$GS_CONDA_ENV")
    fi
    if [[ "$exists" == 1 ]]; then
        echo "conda env $GS_CONDA_ENV already exists: skipping creation"
    else
        echo "creating conda env $GS_CONDA_ENV (python 3.11)"
        conda create -y "${where[@]}" -c conda-forge python=3.11 pip
    fi
    set +u
    conda activate "$GS_CONDA_ENV"
    set -u
else
    if [[ -f "$GS_VENV_DIR/bin/activate" ]]; then
        echo "venv $GS_VENV_DIR already exists: skipping creation"
    else
        command -v python3 >/dev/null 2>&1 || geosense_die "no conda and no python3: load a python module (GS_MODULES)"
        python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
            || geosense_die "python3 >= 3.10 needed for the venv; load a newer python module (GS_MODULES)"
        echo "creating venv $GS_VENV_DIR"
        mkdir -p "$(dirname "$GS_VENV_DIR")"
        python3 -m venv "$GS_VENV_DIR"
    fi
    set +u
    # shellcheck source=/dev/null
    source "$GS_VENV_DIR/bin/activate"
    set -u
fi

echo "python: $(command -v python) ($(python --version 2>&1))"
python -m pip install --no-cache-dir --upgrade pip
if [[ -n "$GS_TORCH_INDEX_URL" ]]; then
    python -m pip install --no-cache-dir torch torchvision --index-url "$GS_TORCH_INDEX_URL"
fi
# the package + test/notebook tools, plus peft (LoRA) and kagglehub (dataset download)
python -m pip install --no-cache-dir -e "$GEOSENSE_REPO[dev]" peft kagglehub

echo
echo "=== check (no GPU on the login node is normal: cuda available = False here) ==="
python - <<'EOF'
import peft, torch, torchvision, transformers
import geosense
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
print("torchvision", torchvision.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("geosense from", geosense.__file__)
EOF
echo "OK: environment ready ($kind)."
