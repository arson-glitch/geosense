#!/usr/bin/env bash
# Submit a GeoSense job with the resources of cluster.env.
#
#     scripts/slurm/submit.sh <run_name> [geosense.finetune args...]       # training (finetune.sbatch)
#     scripts/slurm/submit.sh --smoke [args...]                             # first test (smoke.sbatch)
#     scripts/slurm/submit.sh --export <run_name> [args...]                 # val export (export.sbatch)
#     FINAL_TEST=1 scripts/slurm/submit.sh --export <run_name> [args...]    # val + TEST, contract files
#     DRY_RUN=1 scripts/slurm/submit.sh ...                                 # only print the sbatch command
#
# Why a wrapper: #SBATCH lines cannot read a file, so the partition / account / gres / time of
# cluster.env are turned into sbatch command-line options here (they override the #SBATCH defaults).
# Empty values in cluster.env are not passed at all (SLURM then uses your defaults).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
geosense_load_env

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }

mode=train
case "${1:-}" in
    --smoke) mode=smoke; shift ;;
    --export) mode=export; shift ;;
    ""|-h|--help) usage ;;
esac

case "$mode" in
    smoke)
        script="$GEOSENSE_SLURM_DIR/smoke.sbatch"; job_name=geosense-smoke; time="$GS_SMOKE_TIME" ;;
    export)
        [[ $# -ge 1 && "$1" != -* ]] || usage
        script="$GEOSENSE_SLURM_DIR/export.sbatch"; job_name="export-$1"; time="$GS_EXPORT_TIME" ;;
    train)
        [[ "$1" != -* ]] || usage
        script="$GEOSENSE_SLURM_DIR/finetune.sbatch"; job_name="$1"; time="$GS_TIME" ;;
esac

# Variables the job needs, passed explicitly (some clusters do not export the environment).
export_vars="ALL,GEOSENSE_REPO=$GEOSENSE_REPO,GEOSENSE_ENV_FILE=$GEOSENSE_ENV_FILE"
if [[ "$mode" == export && "${FINAL_TEST:-0}" == 1 ]]; then
    export_vars+=",FINAL_TEST=1"
fi

cmd=(sbatch --job-name="$job_name" --chdir="$GEOSENSE_REPO" --output="$GEOSENSE_REPO/logs/%x-%j.out"
     --export="$export_vars")
[[ -z "$GS_PARTITION" ]] || cmd+=(--partition="$GS_PARTITION")
[[ -z "$GS_ACCOUNT" ]] || cmd+=(--account="$GS_ACCOUNT")
[[ -z "$GS_QOS" ]] || cmd+=(--qos="$GS_QOS")
[[ -z "$GS_GRES" ]] || cmd+=(--gres="$GS_GRES")
cmd+=(--time="$time" --cpus-per-task="$GS_CPUS" --mem="$GS_MEM")
if [[ "$mode" == train && "$GS_AUTO_REQUEUE" == 1 ]]; then
    # 120 s before the limit SLURM sends USR1 to the batch shell, which requeues the job
    cmd+=(--requeue --signal=B:USR1@120)
fi
# (the ${a[@]+...} form keeps an empty array valid under `set -u` in old bash versions)
cmd+=(${GS_EXTRA_SBATCH_ARGS[@]+"${GS_EXTRA_SBATCH_ARGS[@]}"} "$script" "$@")

if [[ "$mode" == export && "${FINAL_TEST:-0}" == 1 ]]; then
    echo "##########################################################################"
    echo "# WARNING: FINAL_TEST=1 exports the TEST set and overwrites the team files"
    echo "#   predictions/streetclip_finetuned_{val,test}_logits.npy"
    echo "# Team rule: test is used ONCE, at the end, for the run chosen on validation."
    echo "##########################################################################"
    if [[ "${DRY_RUN:-0}" != 1 && -t 0 ]]; then
        read -rp "Type the run name ($1) to confirm: " answer
        [[ "$answer" == "$1" ]] || geosense_die "not confirmed, nothing submitted"
    fi
fi

echo "$(printf '%q ' "${cmd[@]}")"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "(DRY_RUN=1: not submitted)"
    exit 0
fi
mkdir -p "$GEOSENSE_REPO/logs"   # SLURM does not create the log folder, and fails silently without it
"${cmd[@]}"
echo "follow it with:  squeue -u \$USER   and   tail -f logs/${job_name}-<jobid>.out"
