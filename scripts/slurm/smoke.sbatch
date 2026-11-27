#!/usr/bin/env bash
# FIRST job to run on the cluster (<= 15 min). It proves, in this order, that:
#   GPU visible -> data path right -> StreetCLIP loads offline -> training runs -> checkpoints are
#   written -> a second run with --resume continues from last.pt instead of starting again.
# It also says whether compute nodes have internet, and logs GPU memory (logs/<job>-gpu.csv)
# to replace the estimates in docs/cluster.md by real numbers.
#
#     scripts/slurm/submit.sh --smoke [extra geosense.finetune args, e.g. --batch-size 16]
#
# Pass 1: --subset 500 --epochs 1 --method last_blocks --unfreeze-blocks 1   (fresh run)
# Pass 2: the same command with --epochs 2 --resume: must skip epoch 1 and train only epoch 2.
#
#SBATCH --job-name=geosense-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
set -euo pipefail

source "${GEOSENSE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
geosense_load_env
geosense_activate
cd "$GEOSENSE_REPO"
mkdir -p logs "$GEOSENSE_CKPT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

geosense_job_header

echo
echo "--- internet from this COMPUTE node (information only) ---"
for url in https://huggingface.co https://www.kaggle.com; do
    if curl -sSI --max-time 10 "$url" >/dev/null 2>&1; then echo "$url: reachable"; else echo "$url: NOT reachable"; fi
done

# A new run name per job: no old checkpoint can make the test pass by accident.
RUN="smoke_${SLURM_JOB_ID:-local}"
workers=$(( ${SLURM_CPUS_PER_TASK:-8} > 1 ? ${SLURM_CPUS_PER_TASK:-8} - 1 : 1 ))
base=(python -m geosense.finetune --root "$GEOSENSE_DATA" --run-name "$RUN" --ckpt-dir "$GEOSENSE_CKPT"
      --subset 500 --method last_blocks --unfreeze-blocks 1 --workers "$workers" "$@")

geosense_start_gpu_log
t0=$SECONDS
echo
echo "=== pass 1 (fresh): $(printf '%q ' "${base[@]}" --epochs 1)"
"${base[@]}" --epochs 1
echo "pass 1 took $(( SECONDS - t0 )) s"

t1=$SECONDS
echo
echo "=== pass 2 (resume): $(printf '%q ' "${base[@]}" --epochs 2 --resume)"
"${base[@]}" --epochs 2 --resume
echo "pass 2 took $(( SECONDS - t1 )) s"
geosense_stop_gpu_log

echo
echo "=== results ==="
# subset runs add a _subset<N> suffix to the history name: glob instead of guessing it
ls -la "$GEOSENSE_CKPT"/"$RUN"*/ || true
shopt -s nullglob
hist=(results/"$RUN"*_history.csv)
ckpt=("$GEOSENSE_CKPT"/"$RUN"*/last.pt)
ok=1
if [[ ${#ckpt[@]} -eq 0 ]]; then echo "FAIL: no last.pt under $GEOSENSE_CKPT/$RUN*"; ok=0; fi
if [[ ${#hist[@]} -eq 0 ]]; then
    echo "FAIL: no results/$RUN*_history.csv"; ok=0
else
    echo "history: ${hist[0]}"
    cat "${hist[0]}"
    # expected epochs 0 (zero-shot reference), 1 (pass 1), 2 (pass 2) - each exactly once
    epochs="$(python -c 'import sys, pandas as pd; print(",".join(map(str, pd.read_csv(sys.argv[1])["epoch"])))' "${hist[0]}")"
    if [[ "$epochs" == "0,1,2" ]]; then echo "PASS: history epochs = $epochs (resume continued at epoch 2)"
    else echo "FAIL: history epochs = $epochs, expected 0,1,2"; ok=0; fi
fi
echo "total smoke time: $(( SECONDS - t0 )) s"
if [[ "$ok" == 1 ]]; then
    echo "SMOKE TEST PASSED. Note the per-pass times and the peak GPU memory above in docs/cluster.md."
    echo "The smoke checkpoints in $GEOSENSE_CKPT/$RUN* can be deleted."
else
    echo "SMOKE TEST FAILED (see above)."
    exit 1
fi
