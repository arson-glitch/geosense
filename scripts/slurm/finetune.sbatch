#!/usr/bin/env bash
# One StreetCLIP fine-tuning run on one GPU.
#
# Submit it with the wrapper (reads partition / account / gres / time from cluster.env):
#     scripts/slurm/submit.sh <run_name> [geosense.finetune args...]
#     scripts/slurm/submit.sh lb2 --method last_blocks --unfreeze-blocks 2
# or directly from the repo root (then the #SBATCH defaults below are used; mkdir -p logs first):
#     sbatch [sbatch opts] scripts/slurm/finetune.sbatch <run_name> [geosense.finetune args...]
#
# The job ALWAYS passes --resume: if the job hit its time limit (or was requeued), submitting the
# SAME command again continues from <GEOSENSE_CKPT>/<run_name>/last.pt (the last finished epoch).
#
#SBATCH --job-name=geosense-ft
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
    echo "usage: sbatch scripts/slurm/finetune.sbatch <run_name> [geosense.finetune args...]" >&2
    exit 2
fi
RUN_NAME="$1"
shift

# Under SLURM this file is a copy in a spool folder, so find the repo from the submit directory.
source "${GEOSENSE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
geosense_load_env
geosense_activate
cd "$GEOSENSE_REPO"
mkdir -p logs "$GEOSENSE_CKPT"

# Compute nodes often have no internet: use only the StreetCLIP files cached by fetch_assets.sh.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1   # log lines appear immediately in logs/<job>.out

geosense_job_header

workers=$(( ${SLURM_CPUS_PER_TASK:-8} > 1 ? ${SLURM_CPUS_PER_TASK:-8} - 1 : 1 ))
# --workers first so that a --workers in the user args wins; --resume and --ckpt-dir always set.
cmd=(python -m geosense.finetune --root "$GEOSENSE_DATA" --run-name "$RUN_NAME" --workers "$workers"
     "$@" --ckpt-dir "$GEOSENSE_CKPT" --resume)
echo
echo "command: $(printf '%q ' "${cmd[@]}")"
echo

geosense_start_gpu_log
status=0
geosense_run_requeueable "${cmd[@]}" || status=$?
geosense_stop_gpu_log
echo "=== finished with exit code $status ($(date '+%Y-%m-%d %H:%M:%S')) ==="
exit "$status"
