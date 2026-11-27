#!/usr/bin/env bash
# Read-only survey of the cluster, run on the LOGIN node:
#     bash scripts/slurm/00_inspect_cluster.sh 2>&1 | tee inspect.txt
# then send inspect.txt to the team. It changes nothing: it only lists partitions, accounts,
# modules, disks and tests internet access. Every command may be missing on this cluster; that
# is printed and the script goes on. Use the output to fill scripts/slurm/cluster.env.
set -euo pipefail

section() { printf '\n==================== [%s] ====================\n' "$1"; }

# run CMD...: print it, run it with a time limit, never fail.
run() {
    echo "\$ $*"
    if ! command -v "$1" >/dev/null 2>&1; then echo "($1: not available)"; return 0; fi
    if command -v timeout >/dev/null 2>&1; then
        timeout 60 "$@" 2>&1 || echo "(exit code $?)"
    else
        "$@" 2>&1 || echo "(exit code $?)"
    fi
}

section "host"
run hostname
run date
run uname -a
echo "user: ${USER:-?}   shell: ${SHELL:-?}"
[[ -f /etc/os-release ]] && grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release || true

section "sinfo"
# partition, availability, time limit, nodes, GRES (GPUs), CPUs/node, memory/node (MB), features, state
run sinfo -o "%20P %6a %12l %6D %30G %6c %10m %25f %T"

section "sinfo GRES per partition"
run sinfo -h -o "%P %G"

section "sinfo per node (first 60 lines)"
echo "\$ sinfo -N -o '%N %P %G %c %m %T' | head -60"
{ sinfo -N -o "%N %P %G %c %m %T" 2>&1 || echo "(sinfo not available)"; } | head -60 || true

section "scontrol show partition"
run scontrol show partition

section "sacctmgr assoc"
run sacctmgr -n -P show assoc user="${USER:-}" format=cluster,account,user,partition,qos,defaultqos,maxwall,grptres,maxtres

section "sacctmgr qos"
run sacctmgr -n -P show qos format=name,priority,maxwall,maxtrespu,maxjobspu,maxsubmitpu

section "my jobs (squeue)"
run squeue -u "${USER:-}"

section "slurm defaults"
run scontrol show config
echo "(SBATCH_*/SLURM_* variables already set in this shell:)"
env | grep -E '^(SBATCH|SLURM)_' || echo "(none)"

section "module avail (conda / python / cuda / gpu lines)"
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        # shellcheck source=/dev/null
        if [[ -f "$f" ]]; then source "$f" >/dev/null 2>&1 || true; break; fi
    done
fi
if type module >/dev/null 2>&1; then
    echo "\$ module avail 2>&1 | grep -iE 'conda|mamba|forge|python|cuda|cudnn|nvidia|gpu|pytorch|torch'"
    { module avail 2>&1 || true; } | grep -iE 'conda|mamba|forge|python|cuda|cudnn|nvidia|gpu|pytorch|torch' || echo "(no matching module)"
    echo "\$ module list"
    { module list 2>&1 || true; }
else
    echo "(no module system found)"
fi

section "nvidia-smi (login node; usually absent - the GPUs are on compute nodes)"
run nvidia-smi

section "python / conda"
for c in python3 python conda mamba micromamba pip3 git git-lfs; do
    if command -v "$c" >/dev/null 2>&1; then
        echo "$c: $(command -v "$c")  ->  $("$c" --version 2>&1 | head -1)"
    else
        echo "$c: not found"
    fi
done
[[ -d "$HOME/miniforge3" || -d "$HOME/miniconda3" || -d "$HOME/anaconda3" ]] \
    && ls -d "$HOME"/miniforge3 "$HOME"/miniconda3 "$HOME"/anaconda3 2>/dev/null || true
run conda env list

section "disk: candidate locations and free space"
for d in "$HOME" "${SCRATCH:-}" "/scratch/${USER:-}" "/scratch" "${WORK:-}" "${PROJECT:-}" "${TMPDIR:-}" "/tmp"; do
    [[ -n "$d" ]] || continue
    if [[ -d "$d" ]]; then
        echo "--- $d (exists, writable: $([[ -w "$d" ]] && echo yes || echo no))"
        df -h "$d" 2>&1 | tail -n +1 || true
    else
        echo "--- $d (does not exist)"
    fi
done
echo "SCRATCH=${SCRATCH:-<unset>}  WORK=${WORK:-<unset>}  PROJECT=${PROJECT:-<unset>}  TMPDIR=${TMPDIR:-<unset>}"
echo "(any *scratch* / *work* paths mounted:)"
{ df -h 2>/dev/null || true; } | grep -iE 'scratch|work|project|data|home' || echo "(none found by name)"

section "disk quotas"
run quota -s
if command -v lfs >/dev/null 2>&1; then
    for d in "$HOME" "${SCRATCH:-}" "/scratch/${USER:-}"; do
        if [[ -n "$d" && -d "$d" ]]; then run lfs quota -h -u "${USER:-}" "$d"; fi
    done
fi
echo "\$ du -sh \$HOME (may take a while)"
{ timeout 60 du -sh "$HOME" 2>/dev/null || echo "(du skipped or timed out)"; } || true

section "internet from the LOGIN node"
echo "(any 'HTTP/...' line = reachable, whatever the code: kaggle.com answers HEAD requests with 404)"
for url in https://huggingface.co https://www.kaggle.com https://pypi.org/simple/pip/ https://download.pytorch.org/whl/; do
    echo "\$ curl -sSI --max-time 10 $url"
    if command -v curl >/dev/null 2>&1; then
        { curl -sSI --max-time 10 "$url" 2>&1 || echo "(no access: exit code $?)"; } | head -1
    else
        echo "(curl not available)"
    fi
done
echo "proxy variables: http_proxy=${http_proxy:-<unset>} https_proxy=${https_proxy:-<unset>}"
echo "(compute-node internet is tested by the smoke job; see its log)"

section "done"
echo "Nothing was changed. Send this whole output to the team."
