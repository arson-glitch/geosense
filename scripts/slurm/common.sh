#!/usr/bin/env bash
# Shared helpers for the GeoSense SLURM scripts. Sourced, never run directly:
#     source "<repo>/scripts/slurm/common.sh"
#
# Why one file: every site-specific value lives in scripts/slurm/cluster.env, and every script
# reads it the same way (find the repo, read cluster.env, load modules, activate the Python env).

GEOSENSE_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

geosense_die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Read cluster.env (or the file named by GEOSENSE_ENV_FILE) and check the required values.
geosense_load_env() {
    local env_file="${GEOSENSE_ENV_FILE:-$GEOSENSE_SLURM_DIR/cluster.env}"
    [[ -f "$env_file" ]] || geosense_die "$env_file not found.
  Copy the template and fill it in:  cp scripts/slurm/cluster.env.example scripts/slurm/cluster.env"
    # shellcheck source=/dev/null
    source "$env_file"
    GEOSENSE_ENV_FILE="$env_file"

    # Defaults for optional values (an empty value means "do not pass it").
    GEOSENSE_REPO="${GEOSENSE_REPO:-$(cd "$GEOSENSE_SLURM_DIR/../.." && pwd)}"
    GS_PARTITION="${GS_PARTITION:-}"
    GS_ACCOUNT="${GS_ACCOUNT:-}"
    GS_QOS="${GS_QOS:-}"
    GS_GRES="${GS_GRES:-gpu:1}"
    GS_TIME="${GS_TIME:-08:00:00}"
    GS_SMOKE_TIME="${GS_SMOKE_TIME:-00:15:00}"
    GS_EXPORT_TIME="${GS_EXPORT_TIME:-01:00:00}"
    GS_CPUS="${GS_CPUS:-8}"
    GS_MEM="${GS_MEM:-48G}"
    GS_AUTO_REQUEUE="${GS_AUTO_REQUEUE:-0}"
    GS_MODULES="${GS_MODULES:-}"
    GS_CONDA_SH="${GS_CONDA_SH:-}"
    GS_CONDA_ENV="${GS_CONDA_ENV:-geosense}"
    GS_VENV_DIR="${GS_VENV_DIR:-$GEOSENSE_REPO/.venv}"
    GS_ENV_KIND="${GS_ENV_KIND:-auto}"
    GS_TORCH_INDEX_URL="${GS_TORCH_INDEX_URL:-}"
    if ! declare -p GS_EXTRA_SBATCH_ARGS >/dev/null 2>&1; then GS_EXTRA_SBATCH_ARGS=(); fi

    [[ -n "${GEOSENSE_DATA:-}" ]] || geosense_die "GEOSENSE_DATA is empty in $env_file"
    [[ -n "${GEOSENSE_CKPT:-}" ]] || geosense_die "GEOSENSE_CKPT is empty in $env_file"
    [[ -n "${HF_HOME:-}" ]] || geosense_die "HF_HOME is empty in $env_file"
    export GEOSENSE_REPO GEOSENSE_DATA GEOSENSE_CKPT HF_HOME GEOSENSE_ENV_FILE
}

# Make the `module` command available in a non-interactive shell (Lmod or Environment Modules).
geosense_init_modules() {
    if ! type module >/dev/null 2>&1; then
        local f
        for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
            # shellcheck source=/dev/null
            if [[ -f "$f" ]]; then source "$f"; break; fi
        done
    fi
    type module >/dev/null 2>&1
}

geosense_load_modules() {
    [[ -n "$GS_MODULES" ]] || return 0
    geosense_init_modules || geosense_die "GS_MODULES is set but the 'module' command is not available"
    local m
    # GS_MODULES is a space-separated list, e.g. "anaconda3/2024.02 cuda/12.1"
    for m in $GS_MODULES; do
        echo "module load $m"
        module load "$m"
    done
}

# Make `conda` usable in this shell. Returns 1 if conda cannot be found.
geosense_init_conda() {
    if [[ -n "$GS_CONDA_SH" ]]; then
        # shellcheck source=/dev/null
        source "$GS_CONDA_SH"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        return 1
    fi
    type conda >/dev/null 2>&1
}

# "conda" or "venv", following GS_ENV_KIND (auto = conda if available).
geosense_env_kind() {
    case "$GS_ENV_KIND" in
        conda|venv) echo "$GS_ENV_KIND" ;;
        auto) if geosense_init_conda; then echo conda; else echo venv; fi ;;
        *) geosense_die "GS_ENV_KIND must be auto, conda or venv (got $GS_ENV_KIND)" ;;
    esac
}

# Load modules and activate the project environment created by setup_env.sh.
geosense_activate() {
    geosense_load_modules
    local kind
    kind="$(geosense_env_kind)"
    if [[ "$kind" == conda ]]; then
        geosense_init_conda || geosense_die "conda not found (set GS_CONDA_SH or GS_MODULES)"
        set +u  # conda's activate scripts use unset variables
        conda activate "$GS_CONDA_ENV" || geosense_die "conda env $GS_CONDA_ENV missing: run scripts/slurm/setup_env.sh"
        set -u
    else
        [[ -f "$GS_VENV_DIR/bin/activate" ]] || geosense_die "venv $GS_VENV_DIR missing: run scripts/slurm/setup_env.sh"
        set +u
        # shellcheck source=/dev/null
        source "$GS_VENV_DIR/bin/activate"
        set -u
    fi
    echo "python: $(command -v python)"
}

# Common header for every job log: where, what, which GPU, which commit.
geosense_job_header() {
    echo "=== GeoSense job ${SLURM_JOB_ID:-local} ($(date '+%Y-%m-%d %H:%M:%S')) ==="
    echo "host: $(hostname)   partition: ${SLURM_JOB_PARTITION:-?}   gpus: ${CUDA_VISIBLE_DEVICES:-?}"
    echo "repo: $GEOSENSE_REPO"
    echo "git commit: $(git -C "$GEOSENSE_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
         "$(git -C "$GEOSENSE_REPO" diff --quiet 2>/dev/null || echo '(uncommitted changes!)')"
    echo "data: $GEOSENSE_DATA"
    echo "checkpoints: $GEOSENSE_CKPT"
    echo "HF_HOME: $HF_HOME   HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-0}"
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; else echo "(nvidia-smi not available)"; fi
    python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
}

# Log GPU memory / utilisation every 10 s to logs/<job>-gpu.csv (the real memory measurement).
geosense_start_gpu_log() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    mkdir -p "$GEOSENSE_REPO/logs"
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 \
        > "$GEOSENSE_REPO/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-local}-gpu.csv" 2>/dev/null &
    GEOSENSE_GPU_LOG_PID=$!
}

geosense_stop_gpu_log() {
    if [[ -n "${GEOSENSE_GPU_LOG_PID:-}" ]]; then kill "$GEOSENSE_GPU_LOG_PID" 2>/dev/null || true; fi
    local f="$GEOSENSE_REPO/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-local}-gpu.csv"
    if [[ -f "$f" ]]; then
        echo "peak GPU memory used (MiB): $(awk -F', ' 'NR>1 {gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$f")"
    fi
}

# Run a command in the background and wait, so that a SIGUSR1 sent by SLURM before the time
# limit (submit.sh adds --signal=B:USR1@120 when GS_AUTO_REQUEUE=1) can requeue the job.
# The requeued job runs the same command; finetune --resume continues from last.pt.
geosense_run_requeueable() {
    if [[ "${GS_AUTO_REQUEUE:-0}" == 1 && -n "${SLURM_JOB_ID:-}" ]]; then
        trap 'echo "SIGUSR1: time limit close, requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1
    fi
    "$@" &
    local pid=$! status=0
    wait "$pid" || status=$?
    # wait returns >128 when interrupted by the trapped signal: wait again for the real exit code
    while kill -0 "$pid" 2>/dev/null; do wait "$pid" || status=$?; done
    return "$status"
}
