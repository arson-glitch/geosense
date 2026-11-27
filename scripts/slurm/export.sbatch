#!/usr/bin/env bash
# Export logits from <GEOSENSE_CKPT>/<run_name>/best.pt, without training.
#
#     scripts/slurm/submit.sh --export <run_name> [--final]   (model settings are read from best.pt)
#         -> predictions/runs/<run_name>_val_logits.npy        (validation only, for comparisons)
#
#     FINAL_TEST=1 scripts/slurm/submit.sh --export <run_name> [args...]
#         -> predictions/streetclip_finetuned_{val,test}_logits.npy   (the TEAM CONTRACT files)
#
# !!! FINAL_TEST=1 touches the TEST set. Team rule: the test set is used ONCE, at the very end,
# !!! for the single run chosen on VALIDATION. Never use it to choose between runs.
#
#SBATCH --job-name=geosense-export
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
    echo "usage: [FINAL_TEST=1] sbatch scripts/slurm/export.sbatch <run_name> [geosense.finetune args...]" >&2
    exit 2
fi
RUN_NAME="$1"
shift

source "${GEOSENSE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
geosense_load_env
geosense_activate
cd "$GEOSENSE_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

geosense_job_header

workers=$(( ${SLURM_CPUS_PER_TASK:-8} > 1 ? ${SLURM_CPUS_PER_TASK:-8} - 1 : 1 ))
cmd=(python -m geosense.finetune --root "$GEOSENSE_DATA" --run-name "$RUN_NAME" --workers "$workers"
     "$@" --ckpt-dir "$GEOSENSE_CKPT" --export-only)
if [[ "${FINAL_TEST:-0}" == 1 ]]; then
    echo "##########################################################################"
    echo "# FINAL_TEST=1: exporting the TEST set and writing the team contract files"
    echo "#   predictions/streetclip_finetuned_val_logits.npy"
    echo "#   predictions/streetclip_finetuned_test_logits.npy"
    echo "# Only for the ONE run chosen on validation, once, at the end."
    echo "##########################################################################"
    cmd+=(--export val,test --final)
else
    cmd+=(--export val)
fi
echo
echo "command: $(printf '%q ' "${cmd[@]}")"
echo
status=0
"${cmd[@]}" || status=$?
echo "=== finished with exit code $status ($(date '+%Y-%m-%d %H:%M:%S')) ==="
ls -la predictions/runs/"$RUN_NAME"_* predictions/streetclip_finetuned_* 2>/dev/null || true
exit "$status"
